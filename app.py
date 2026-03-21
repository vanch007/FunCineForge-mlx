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
_MODEL = None
_KWARGS = {}
_DEMO_ITEMS = []

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
    global _MODEL, _KWARGS
    if _MODEL is not None:
        return "✅ Models already loaded"

    os.chdir(EXP_DIR)
    device = get_device()
    progress(0.0, desc="Loading config...")

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(DECODE_YAML)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    lm_ckpt = os.path.join(CKPT_DIR, "llm", "ds-model.pt.best", "mp_rank_00_model_states.pt")
    fm_ckpt = os.path.join(CKPT_DIR, "flow", "ds-model.pt.best", "mp_rank_00_model_states.pt")
    voc_ckpt = os.path.join(CKPT_DIR, "vocoder", "ds-model.pt.best", "avg_5_removewn.pt")

    progress(0.1, desc="Loading LLM (Qwen2)...")
    lm_exp_dir, lm_model_name, _, _ = lm_ckpt.rsplit("/", 3)
    lm_model = AutoModel(
        model=os.path.join(lm_exp_dir, lm_model_name),
        init_param=lm_ckpt, output_dir=None, device=device,
    )
    lm_model.model.to(dtype_map[cfg.get("llm_dtype", "fp32")])

    progress(0.4, desc="Loading Flow Matching...")
    fm_exp_dir, fm_model_name, _, _ = fm_ckpt.rsplit("/", 3)
    fm_model = AutoModel(
        model=os.path.join(fm_exp_dir, fm_model_name),
        init_param=fm_ckpt, output_dir=None, device=device,
    )
    fm_model.model.to(dtype_map[cfg.get("fm_dtype", "fp32")])

    progress(0.7, desc="Loading Vocoder (HiFi-GAN)...")
    voc_exp_dir, voc_model_name, _, _ = voc_ckpt.rsplit("/", 3)
    voc_model = AutoModel(
        model=os.path.join(voc_exp_dir, voc_model_name),
        init_param=voc_ckpt, output_dir=None, device=device,
    )
    voc_model.model.to(dtype_map[cfg.get("voc_dtype", "fp32")])

    progress(0.9, desc="Building inference pipeline...")
    cfg["output_dir"] = OUTPUT_DIR
    cfg["tokenizer"] = None
    cfg["lm_ckpt_path"] = lm_ckpt
    cfg["fm_ckpt_path"] = fm_ckpt
    cfg["voc_ckpt_path"] = voc_ckpt
    infer = AutoModel(
        **cfg, lm_model=lm_model, fm_model=fm_model, voc_model=voc_model,
    )
    _MODEL = infer
    _KWARGS = cfg

    progress(1.0, desc="Done!")
    return f"✅ Models loaded on {device} | MLX: {cfg.get('use_mlx', False)}"


# ──────────────────────────────────────────────
# Demo data
# ──────────────────────────────────────────────
def load_demo_items():
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
    items = load_demo_items()
    return [[i, it["utt"], it["type"],
             it["text"][:80] + ("..." if len(it["text"]) > 80 else "")]
            for i, it in enumerate(items)]


def resolve_path(rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(EXP_DIR, rel_path)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def run_inference(data_item, seed=0, min_len=50, max_len=1500, sampling="ras"):
    if _MODEL is None:
        raise gr.Error("⚠️ Please load models first!")

    kwargs = dict(_KWARGS)
    kwargs["random_seed"] = seed
    kwargs["min_length"] = min_len
    kwargs["max_length"] = max_len
    kwargs["sampling"] = sampling
    kwargs["output_dir"] = OUTPUT_DIR

    index_ds_class = tables.index_ds_classes.get(kwargs.get("index_ds"))
    dataset_conf = kwargs.get("dataset_conf", {})

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=OUTPUT_DIR
    ) as f:
        json.dump(data_item, f, ensure_ascii=False)
        f.write("\n")
        tmp_jsonl = f.name

    try:
        index_ds = index_ds_class(tmp_jsonl, **dataset_conf)
        if len(index_ds) == 0:
            raise gr.Error("⚠️ Data item filtered — check text/speech length")
        _MODEL.inference(input=index_ds, input_len=len(index_ds))
    finally:
        os.unlink(tmp_jsonl)

    utt = data_item["utt"]
    wav_path = os.path.join(OUTPUT_DIR, "wav", f"{utt}.wav")
    video_path = os.path.join(OUTPUT_DIR, "mp4", f"{utt}.mp4")
    return (
        wav_path if os.path.exists(wav_path) else None,
        video_path if os.path.exists(video_path) else None,
    )


def build_jsonl_item(utt, text, clue, scene_type, vocal_path, video_path,
                     face_path, dialogue, speech_length):
    messages = [
        {"role": "text", "content": text},
        {"role": "vocal", "content": vocal_path},
        {"role": "video", "content": video_path or ""},
        {"role": "face", "content": face_path},
        {"role": "dialogue", "content": dialogue},
        {"role": "clue", "content": clue},
    ]
    return {
        "messages": messages, "utt": utt,
        "type": scene_type, "speech_length": speech_length,
    }


def create_empty_face_pkl(speech_length, face_size=512):
    pkl_path = os.path.join(OUTPUT_DIR, f"_tmp_face_{int(time.time())}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "embeddings": np.zeros((1, face_size), dtype=np.float32),
            "faceI": np.array([0], dtype=np.int64),
        }, f)
    return pkl_path


def estimate_speech_length(text, rate=5.0):
    return max(50, min(1500, int(len(text) * rate)))


# ──────────────────────────────────────────────
# Event handlers
# ──────────────────────────────────────────────
TYPE_MAP = {"旁白 Narration": "旁白", "独白 Monologue": "独白",
            "对话 Dialogue": "对话", "多人 Multi-Speaker": "多人"}
TYPE_REV = {v: k for k, v in TYPE_MAP.items()}
GENDER_MAP = {"男 Male": "male", "女 Female": "female"}
GENDER_REV = {"male": "男 Male", "female": "女 Female"}
AGE_MAP = {
    "儿童 Child": "child", "青年 Youth": "teenager",
    "中年 Adult": "adult", "中老年 Middle-aged": "middle-aged",
    "老年 Elderly": "elderly",
}
AGE_REV = {v: k for k, v in AGE_MAP.items()}


def on_demo_select(evt: gr.SelectData):
    """Fill the form fields when a demo item is selected."""
    items = load_demo_items()
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx < 0 or idx >= len(items):
        return [gr.update()] * 8

    item = items[idx]
    vocal = resolve_path(item["vocal"])
    video = resolve_path(item["video"])

    # Derive gender/age from first dialogue entry
    dlg = item["dialogue"]
    gender = GENDER_REV.get(dlg[0]["gender"], "男 Male") if dlg else "男 Male"
    age = AGE_REV.get(dlg[0]["age"], "中年 Adult") if dlg else "中年 Adult"
    scene = TYPE_REV.get(item["type"], "独白 Monologue")

    return (
        item["text"],                       # text
        item["clue"],                       # clue
        scene,                              # scene_type
        gender,                             # gender
        age,                                # age
        vocal if os.path.exists(vocal) else None,  # ref audio
        video if os.path.exists(video) else None,  # ref video
        f"demo:{item['utt']}",             # hidden demo key
    )


def on_synthesize(
    text, clue, scene_type, gender, age,
    vocal_file, video_file, demo_key,
    seed, min_len, max_len, sampling
):
    """Unified synthesis handler — works for both demo and custom.
    
    Demo mode: uses demo reference audio/video/face, but ALWAYS respects user's
    text, clue, scene type, gender, and age from the UI form.
    """
    if not text:
        raise gr.Error("⚠️ Please enter text to synthesize")

    # Check if this is a demo item (for reference audio/video/face only)
    is_demo = demo_key and demo_key.startswith("demo:")
    if is_demo:
        utt = demo_key.split(":", 1)[1]
        items = load_demo_items()
        item = next((it for it in items if it["utt"] == utt), None)
        if item is None:
            raise gr.Error(f"⚠️ Demo item '{utt}' not found")

        # Use demo's reference files but user's text/clue/settings
        vocal_path = resolve_path(item["vocal"])
        video_path = resolve_path(item["video"])
        face_path = resolve_path(item["face"])
        dialogue = item["dialogue"]
        # Recalculate speech_length based on user's text, not demo's original
        speech_length = estimate_speech_length(text)
        utt_id = f"ref_{utt}"

        data = build_jsonl_item(
            utt=utt_id,
            text=text,                                    # User's text, NOT demo's
            clue=clue or item["clue"],                    # User's clue, fallback demo's
            scene_type=TYPE_MAP.get(scene_type, "独白"),   # User's scene type
            vocal_path=vocal_path,
            video_path=video_path,
            face_path=face_path,
            dialogue=dialogue,
            speech_length=speech_length,
        )
    else:
        # Custom mode
        if vocal_file is None:
            raise gr.Error("⚠️ Please upload a reference audio file")

        utt_id = f"custom_{int(time.time())}"
        speech_length = estimate_speech_length(text)

        try:
            info = sf.info(vocal_file)
            audio_duration = info.duration
        except Exception:
            audio_duration = speech_length / 25.0

        dialogue = [{
            "start": 0.0,
            "duration": round(audio_duration, 2),
            "spk": "1",
            "gender": GENDER_MAP.get(gender, "male"),
            "age": AGE_MAP.get(age, "adult"),
        }]

        face_path = create_empty_face_pkl(speech_length)
        video_path = ""
        if video_file is not None:
            import shutil
            video_path = os.path.join(OUTPUT_DIR, f"{utt_id}_input.mp4")
            shutil.copy2(video_file, video_path)

        data = build_jsonl_item(
            utt=utt_id, text=text,
            clue=clue or "A speaker speaks clearly with natural emotion.",
            scene_type=TYPE_MAP.get(scene_type, "独白"),
            vocal_path=vocal_file, video_path=video_path,
            face_path=face_path, dialogue=dialogue,
            speech_length=speech_length,
        )

    t0 = time.time()
    wav_path, vid_path = run_inference(
        data, seed=int(seed), min_len=int(min_len),
        max_len=int(max_len), sampling=sampling
    )
    elapsed = time.time() - t0

    # Clean up temp face pkl (custom mode only)
    if not is_demo:
        try:
            os.unlink(face_path)
        except Exception:
            pass

    return wav_path, vid_path, f"✅ Done in {elapsed:.1f}s"


def on_clear_demo():
    """Clear the demo selection — reset to custom mode."""
    return (
        "",     # text
        "",     # clue
        "独白 Monologue",  # scene_type
        "男 Male",         # gender
        "中年 Adult",      # age
        None,   # ref audio
        None,   # ref video
        "",     # demo_key (empty = custom mode)
    )


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="FunCineForge MLX — Movie Dubbing",
    ) as app:
        gr.Markdown(
            "# 🎬 FunCineForge MLX WebUI\n"
            "*Zero-Shot Movie Dubbing — MLX-accelerated on Apple Silicon*",
        )

        # ── Model loader ──
        with gr.Row():
            load_btn = gr.Button("🚀 Load Models", variant="primary", scale=2)
            load_status = gr.Textbox(
                value="⏳ Models not loaded", label="Status",
                interactive=False, scale=3
            )
        load_btn.click(load_models, outputs=load_status)

        # Hidden state for demo item tracking
        demo_key = gr.Textbox(value="", visible=False, elem_id="demo-key")

        # ── Main input area ──
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Enter text or select a demo below...",
                    lines=4,
                )
                clue = gr.Textbox(
                    label="Scene Description (Clue)",
                    placeholder="e.g. A female speaker with warm, gentle tone...",
                    lines=2,
                )
                with gr.Row():
                    scene_type = gr.Dropdown(
                        list(TYPE_MAP.keys()), value="独白 Monologue",
                        label="Scene Type",
                    )
                    gender = gr.Dropdown(
                        list(GENDER_MAP.keys()), value="男 Male",
                        label="Gender",
                    )
                    age = gr.Dropdown(
                        list(AGE_MAP.keys()), value="中年 Adult",
                        label="Age",
                    )

            with gr.Column(scale=1):
                ref_audio = gr.Audio(
                    label="Reference Audio (voice timbre)", type="filepath",
                )
                ref_video = gr.Video(label="Video to Dub (optional)")

        # ── Synthesize button ──
        synth_btn = gr.Button(
            "🎙️ Synthesize", variant="primary", size="lg",
        )

        # ── Output ──
        with gr.Row():
            out_audio = gr.Audio(label="Generated Audio", type="filepath")
            out_video = gr.Video(label="Generated Video")
        status_text = gr.Markdown("")

        # ── Settings ──
        with gr.Accordion("⚙️ Settings", open=False):
            with gr.Row():
                sampling = gr.Dropdown(
                    ["ras", "top_k", "nucleus"], value="ras",
                    label="Sampling", scale=1,
                )
                seed = gr.Number(value=0, label="Seed", precision=0, scale=1)
                min_len = gr.Number(value=50, label="Min Length", precision=0, scale=1)
                max_len = gr.Number(value=1500, label="Max Length", precision=0, scale=1)

        # ── Demo examples at bottom ──
        with gr.Accordion("📚 Demo Examples — click to fill form", open=False):
            clear_btn = gr.Button("🔄 Clear Selection (Custom Mode)", size="sm")
            demo_table = gr.Dataframe(
                headers=["#", "ID", "Type", "Text"],
                value=get_demo_table(),
                interactive=False,
                wrap=True,
                max_height=300,
            )

        # ── Events ──
        form_outputs = [text, clue, scene_type, gender, age,
                        ref_audio, ref_video, demo_key]

        demo_table.select(on_demo_select, outputs=form_outputs)
        clear_btn.click(on_clear_demo, outputs=form_outputs)

        synth_btn.click(
            on_synthesize,
            inputs=[text, clue, scene_type, gender, age,
                    ref_audio, ref_video, demo_key,
                    seed, min_len, max_len, sampling],
            outputs=[out_audio, out_video, status_text],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
