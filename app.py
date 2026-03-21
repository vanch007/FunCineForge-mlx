#!/usr/bin/env python3
"""FunCineForge Gradio WebUI — MLX-accelerated movie dubbing interface."""

import os
import sys
import json
import time
import logging
import tempfile
import pickle
import numpy as np
import torch
import gradio as gr
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funcineforge import AutoModel
from funcineforge.models.utils import dtype_map
from funcineforge.register import tables
from funcineforge.utils.set_all_random_seed import set_all_random_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webui")

# ──────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────
_MODEL = None  # FunCineForgeInferModel instance
_KWARGS = {}   # merged config kwargs
_DEMO_ITEMS = []  # parsed demo.jsonl

# Paths (relative to exps/)
EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exps")
CKPT_DIR = os.path.join(EXP_DIR, "funcineforge_zh_en")
DECODE_YAML = os.path.join(EXP_DIR, "decode_conf", "decode.yaml")
DEMO_JSONL = os.path.join(EXP_DIR, "data", "demo.jsonl")
OUTPUT_DIR = os.path.join(EXP_DIR, "webui_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def load_models(progress=gr.Progress()):
    """Load LM + FM + VOC → InferModel. Called once at startup."""
    global _MODEL, _KWARGS

    if _MODEL is not None:
        return "✅ Models already loaded"

    # Model configs use relative paths (e.g. ../tokenizer/) that expect CWD=exps/
    os.chdir(EXP_DIR)

    device = get_device()
    progress(0.0, desc="Loading config...")

    # Load hydra config
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(DECODE_YAML)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    lm_ckpt = os.path.join(CKPT_DIR, "llm", "ds-model.pt.best", "mp_rank_00_model_states.pt")
    fm_ckpt = os.path.join(CKPT_DIR, "flow", "ds-model.pt.best", "mp_rank_00_model_states.pt")
    voc_ckpt = os.path.join(CKPT_DIR, "vocoder", "ds-model.pt.best", "avg_5_removewn.pt")

    # LM
    progress(0.1, desc="Loading LLM (Qwen2)...")
    lm_exp_dir, lm_model_name, _, _ = lm_ckpt.rsplit("/", 3)
    lm_model = AutoModel(
        model=os.path.join(lm_exp_dir, lm_model_name),
        init_param=lm_ckpt,
        output_dir=None,
        device=device,
    )
    lm_model.model.to(dtype_map[cfg.get("llm_dtype", "fp32")])

    # FM
    progress(0.4, desc="Loading Flow Matching...")
    fm_exp_dir, fm_model_name, _, _ = fm_ckpt.rsplit("/", 3)
    fm_model = AutoModel(
        model=os.path.join(fm_exp_dir, fm_model_name),
        init_param=fm_ckpt,
        output_dir=None,
        device=device,
    )
    fm_model.model.to(dtype_map[cfg.get("fm_dtype", "fp32")])

    # VOC
    progress(0.7, desc="Loading Vocoder (HiFi-GAN)...")
    voc_exp_dir, voc_model_name, _, _ = voc_ckpt.rsplit("/", 3)
    voc_model = AutoModel(
        model=os.path.join(voc_exp_dir, voc_model_name),
        init_param=voc_ckpt,
        output_dir=None,
        device=device,
    )
    voc_model.model.to(dtype_map[cfg.get("voc_dtype", "fp32")])

    # InferModel
    progress(0.9, desc="Building inference pipeline...")
    cfg["output_dir"] = OUTPUT_DIR
    cfg["tokenizer"] = None
    cfg["lm_ckpt_path"] = lm_ckpt
    cfg["fm_ckpt_path"] = fm_ckpt
    cfg["voc_ckpt_path"] = voc_ckpt
    infer = AutoModel(
        **cfg,
        lm_model=lm_model,
        fm_model=fm_model,
        voc_model=voc_model,
    )
    _MODEL = infer
    _KWARGS = cfg

    progress(1.0, desc="Done!")
    return f"✅ Models loaded on **{device}** | MLX: {cfg.get('use_mlx', False)}"


# ──────────────────────────────────────────────
# Demo JSONL
# ──────────────────────────────────────────────
def load_demo_items():
    """Parse demo.jsonl into a list of dicts for the gallery."""
    global _DEMO_ITEMS
    if _DEMO_ITEMS:
        return _DEMO_ITEMS

    if not os.path.exists(DEMO_JSONL):
        return []

    items = []
    with open(DEMO_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            msgs = {m["role"]: m["content"] for m in d["messages"]}
            items.append({
                "utt": d["utt"],
                "type": d.get("type", "独白"),
                "text": msgs.get("text", ""),
                "clue": msgs.get("clue", ""),
                "vocal": msgs.get("vocal", ""),
                "video": msgs.get("video", ""),
                "face": msgs.get("face", ""),
                "dialogue": msgs.get("dialogue", []),
                "speech_length": d.get("speech_length", 100),
            })
    _DEMO_ITEMS = items
    return items


def get_demo_table():
    """Build a Dataframe for the demo gallery."""
    items = load_demo_items()
    rows = []
    for i, item in enumerate(items):
        rows.append([
            i,
            item["utt"],
            item["type"],
            item["text"][:80] + ("..." if len(item["text"]) > 80 else ""),
        ])
    return rows


def resolve_demo_path(rel_path):
    """Resolve a demo data path relative to exps/."""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(EXP_DIR, rel_path)


# ──────────────────────────────────────────────
# Inference core
# ──────────────────────────────────────────────
def run_inference(data_item, seed=0, min_len=50, max_len=1500, sampling="ras"):
    """Run single-item inference using the loaded model."""
    if _MODEL is None:
        raise gr.Error("⚠️ Please load models first!")

    model = _MODEL.model
    kwargs = dict(_KWARGS)
    kwargs["random_seed"] = seed
    kwargs["min_length"] = min_len
    kwargs["max_length"] = max_len
    kwargs["sampling"] = sampling
    kwargs["output_dir"] = OUTPUT_DIR

    # Build index_ds item
    index_ds_class = tables.index_ds_classes.get(kwargs.get("index_ds"))
    dataset_conf = kwargs.get("dataset_conf", {})

    # Write temp jsonl
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=OUTPUT_DIR
    ) as f:
        json.dump(data_item, f, ensure_ascii=False)
        f.write("\n")
        tmp_jsonl = f.name

    try:
        index_ds = index_ds_class(tmp_jsonl, **dataset_conf)
        if len(index_ds) == 0:
            raise gr.Error("⚠️ Data item was filtered out — check speech_length / text_length")

        _MODEL.inference(input=index_ds, input_len=len(index_ds))
    finally:
        os.unlink(tmp_jsonl)

    # Find output files
    utt = data_item["utt"]
    wav_path = os.path.join(OUTPUT_DIR, "wav", f"{utt}.wav")
    video_path = os.path.join(OUTPUT_DIR, "mp4", f"{utt}.mp4")

    return (
        wav_path if os.path.exists(wav_path) else None,
        video_path if os.path.exists(video_path) else None,
    )


def build_jsonl_item(
    utt, text, clue, scene_type, vocal_path, video_path, face_path,
    dialogue, speech_length
):
    """Build a JSONL-compatible dict for inference."""
    messages = [
        {"role": "text", "content": text},
        {"role": "vocal", "content": vocal_path},
        {"role": "video", "content": video_path or ""},
        {"role": "face", "content": face_path},
        {"role": "dialogue", "content": dialogue},
        {"role": "clue", "content": clue},
    ]
    return {
        "messages": messages,
        "utt": utt,
        "type": scene_type,
        "speech_length": speech_length,
    }


def create_empty_face_pkl(speech_length, face_size=512):
    """Create a face embedding pkl (zeros) for custom inputs without video face data."""
    pkl_path = os.path.join(OUTPUT_DIR, f"_tmp_face_{int(time.time())}.pkl")
    embeddings = np.zeros((1, face_size), dtype=np.float32)
    faceI = np.array([0], dtype=np.int64)
    with open(pkl_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "faceI": faceI}, f)
    return pkl_path


def estimate_speech_length(text, rate=5.0):
    """Estimate speech token count from text length.
    Rough: ~5 tokens per character for Chinese, ~3 per word for English.
    Token rate = 25 fps, so 1 second = 25 tokens.
    A rough estimate: len(text) * rate → clamp to [50, 1500].
    """
    n = len(text)
    estimated = int(n * rate)
    return max(50, min(1500, estimated))


# ──────────────────────────────────────────────
# Tab 1: Demo Gallery handlers
# ──────────────────────────────────────────────
def on_demo_select(evt: gr.SelectData):
    """When user selects a row in the demo table."""
    items = load_demo_items()
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx < 0 or idx >= len(items):
        return "", "", "", None, None

    item = items[idx]
    vocal = resolve_demo_path(item["vocal"])
    video = resolve_demo_path(item["video"])

    return (
        item["text"],
        item["clue"],
        f"{item['type']} | {item['utt']}",
        vocal if os.path.exists(vocal) else None,
        video if os.path.exists(video) else None,
    )


def on_demo_synth(selected_info, seed, min_len, max_len, sampling):
    """Synthesize the selected demo item."""
    if not selected_info:
        raise gr.Error("⚠️ Please select a demo item first")

    # Parse utt from selected_info
    parts = selected_info.split(" | ")
    utt = parts[1] if len(parts) > 1 else parts[0]

    items = load_demo_items()
    item = None
    for it in items:
        if it["utt"] == utt:
            item = it
            break
    if item is None:
        raise gr.Error(f"⚠️ Item '{utt}' not found")

    # Build jsonl item with resolved paths
    data = build_jsonl_item(
        utt=item["utt"],
        text=item["text"],
        clue=item["clue"],
        scene_type=item["type"],
        vocal_path=resolve_demo_path(item["vocal"]),
        video_path=resolve_demo_path(item["video"]),
        face_path=resolve_demo_path(item["face"]),
        dialogue=item["dialogue"],
        speech_length=item["speech_length"],
    )

    t0 = time.time()
    wav_path, video_path = run_inference(
        data, seed=int(seed), min_len=int(min_len),
        max_len=int(max_len), sampling=sampling
    )
    elapsed = time.time() - t0

    status = f"✅ Done in {elapsed:.1f}s"
    return wav_path, video_path, status


# ──────────────────────────────────────────────
# Tab 2: Custom TTS handlers
# ──────────────────────────────────────────────
def on_custom_synth(
    text, clue, scene_type, gender, age,
    vocal_file, video_file, seed, min_len, max_len, sampling
):
    """Synthesize from custom user inputs."""
    if not text:
        raise gr.Error("⚠️ Please enter text to synthesize")
    if vocal_file is None:
        raise gr.Error("⚠️ Please upload a reference audio file")

    utt = f"custom_{int(time.time())}"

    # Estimate speech length from text
    speech_length = estimate_speech_length(text)

    # Get audio duration for dialogue metadata
    try:
        info = sf.info(vocal_file)
        audio_duration = info.duration
    except Exception:
        audio_duration = speech_length / 25.0

    # Build dialogue metadata
    type_map = {"旁白 Narration": "旁白", "独白 Monologue": "独白",
                "对话 Dialogue": "对话", "多人 Multi-Speaker": "多人"}
    gender_map = {"男 Male": "male", "女 Female": "female"}
    age_map = {
        "儿童 Child": "child", "青年 Youth": "teenager",
        "中年 Adult": "adult", "中老年 Middle-aged": "middle-aged",
        "老年 Elderly": "elderly"
    }

    dialogue = [{
        "start": 0.0,
        "duration": round(audio_duration, 2),
        "spk": "1",
        "gender": gender_map.get(gender, "male"),
        "age": age_map.get(age, "adult"),
    }]

    # Create face pkl (zeros for now)
    face_path = create_empty_face_pkl(speech_length)

    # Handle video: copy to output dir if provided
    video_path = ""
    if video_file is not None:
        import shutil
        video_path = os.path.join(OUTPUT_DIR, f"{utt}_input.mp4")
        shutil.copy2(video_file, video_path)

    data = build_jsonl_item(
        utt=utt,
        text=text,
        clue=clue or "A speaker speaks clearly with natural emotion.",
        scene_type=type_map.get(scene_type, "独白"),
        vocal_path=vocal_file,
        video_path=video_path,
        face_path=face_path,
        dialogue=dialogue,
        speech_length=speech_length,
    )

    t0 = time.time()
    wav_path, vid_path = run_inference(
        data, seed=int(seed), min_len=int(min_len),
        max_len=int(max_len), sampling=sampling
    )
    elapsed = time.time() - t0

    # Clean up temp face pkl
    try:
        os.unlink(face_path)
    except Exception:
        pass

    status = f"✅ Done in {elapsed:.1f}s | {speech_length} tokens"
    return wav_path, vid_path, status


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1200px !important; }
.title-text { text-align: center; }
#synth-btn { min-height: 45px; }
"""


def build_ui():
    with gr.Blocks(
        title="FunCineForge MLX — Movie Dubbing",
    ) as app:
        gr.Markdown(
            "# 🎬 FunCineForge MLX WebUI\n"
            "*Zero-Shot Movie Dubbing — MLX-accelerated on Apple Silicon*",
            elem_classes="title-text",
        )

        # Model loader
        with gr.Row():
            load_btn = gr.Button("🚀 Load Models", variant="primary", scale=2)
            load_status = gr.Textbox(value="⏳ Models not loaded", label="Status", interactive=False, scale=3)
        load_btn.click(load_models, outputs=load_status)

        # ── Shared settings (sidebar-style) ──
        with gr.Accordion("⚙️ Settings", open=False):
            with gr.Row():
                sampling = gr.Dropdown(
                    ["ras", "top_k", "nucleus"], value="ras",
                    label="Sampling", scale=1
                )
                seed = gr.Number(value=0, label="Seed", precision=0, scale=1)
                min_len = gr.Number(value=50, label="Min Length", precision=0, scale=1)
                max_len = gr.Number(value=1500, label="Max Length", precision=0, scale=1)

        # ── Tabs ──
        with gr.Tabs():
            # ──────── Tab 1: Demo Gallery ────────
            with gr.Tab("📚 Demo Gallery"):
                gr.Markdown("Select a demo item from the table, then click **Synthesize**.")

                demo_table = gr.Dataframe(
                    headers=["#", "ID", "Type", "Text"],
                    value=get_demo_table(),
                    interactive=False,
                    wrap=True,
                    max_height=300,
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        demo_text = gr.Textbox(label="Text", lines=3, interactive=False)
                        demo_clue = gr.Textbox(label="Scene Clue", lines=2, interactive=False)
                        demo_info = gr.Textbox(
                            label="Selected", interactive=False, max_lines=1
                        )
                    with gr.Column(scale=1):
                        demo_ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                        demo_ref_video = gr.Video(label="Reference Video")

                demo_synth_btn = gr.Button(
                    "🎙️ Synthesize", variant="primary", elem_id="synth-btn"
                )

                with gr.Row():
                    demo_out_audio = gr.Audio(label="Generated Audio", type="filepath")
                    demo_out_video = gr.Video(label="Generated Video")
                demo_status = gr.Markdown("")

                # Events
                demo_table.select(
                    on_demo_select,
                    outputs=[demo_text, demo_clue, demo_info,
                             demo_ref_audio, demo_ref_video],
                )
                demo_synth_btn.click(
                    on_demo_synth,
                    inputs=[demo_info, seed, min_len, max_len, sampling],
                    outputs=[demo_out_audio, demo_out_video, demo_status],
                )

            # ──────── Tab 2: Custom TTS ────────
            with gr.Tab("🎬 Custom Dubbing"):
                gr.Markdown(
                    "Upload a **reference audio** (voice timbre) and optionally a **video** "
                    "to dub. Enter the text you want spoken."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        custom_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Enter the text you want dubbed...",
                            lines=4,
                        )
                        custom_clue = gr.Textbox(
                            label="Scene Description (Clue)",
                            placeholder="e.g. A female speaker with warm, gentle tone...",
                            lines=2,
                        )
                        with gr.Row():
                            custom_type = gr.Dropdown(
                                ["独白 Monologue", "旁白 Narration",
                                 "对话 Dialogue", "多人 Multi-Speaker"],
                                value="独白 Monologue",
                                label="Scene Type",
                            )
                            custom_gender = gr.Dropdown(
                                ["男 Male", "女 Female"],
                                value="男 Male", label="Gender",
                            )
                            custom_age = gr.Dropdown(
                                ["儿童 Child", "青年 Youth", "中年 Adult",
                                 "中老年 Middle-aged", "老年 Elderly"],
                                value="中年 Adult", label="Age",
                            )

                    with gr.Column(scale=1):
                        custom_vocal = gr.Audio(
                            label="Reference Audio (voice timbre)",
                            type="filepath",
                        )
                        custom_video = gr.Video(
                            label="Video to Dub (optional)",
                        )

                custom_synth_btn = gr.Button(
                    "🎙️ Generate Dubbing", variant="primary", elem_id="synth-btn"
                )

                with gr.Row():
                    custom_out_audio = gr.Audio(label="Generated Audio", type="filepath")
                    custom_out_video = gr.Video(label="Dubbed Video")
                custom_status = gr.Markdown("")

                custom_synth_btn.click(
                    on_custom_synth,
                    inputs=[
                        custom_text, custom_clue, custom_type,
                        custom_gender, custom_age,
                        custom_vocal, custom_video,
                        seed, min_len, max_len, sampling,
                    ],
                    outputs=[custom_out_audio, custom_out_video, custom_status],
                )

    return app


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
