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
# Inference helpers
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
# Constants & mappings
# ──────────────────────────────────────────────
TYPE_MAP = {"旁白 Narration": "旁白", "独白 Monologue": "独白",
            "对话 Dialogue": "对话", "多人 Multi-Speaker": "多人"}
TYPE_REV = {v: k for k, v in TYPE_MAP.items()}
GENDER_MAP = {"男 Male": "male", "女 Female": "female"}
GENDER_REV = {"male": "男 Male", "female": "女 Female",
              "男": "男 Male", "女": "女 Female"}
AGE_MAP = {
    "儿童 Child": "child", "青年 Youth": "teenager",
    "中年 Adult": "adult", "中老年 Middle-aged": "middle-aged",
    "老年 Elderly": "elderly",
}
AGE_REV = {v: k for k, v in AGE_MAP.items()}
# Extra Chinese→display mappings
AGE_REV.update({"青年": "青年 Youth", "中年": "中年 Adult",
                "儿童": "儿童 Child", "中老年": "中老年 Middle-aged",
                "老年": "老年 Elderly"})

DIALOGUE_TYPES = {"对话 Dialogue", "多人 Multi-Speaker"}

# Default dialogue table (2 speakers)
DEFAULT_DIALOGUE_TABLE = [
    ["1", "男 Male", "中年 Adult", 0.0, 3.0],
    ["2", "女 Female", "中年 Adult", 3.0, 3.0],
]

CAMPP_MODEL_PATH = os.path.join(CKPT_DIR, "camplus.onnx")


def auto_detect_dialogue(audio_path, min_speech_dur=0.3, min_silence_dur=0.4,
                          energy_threshold=0.01):
    """Auto-detect dialogue segments from audio file.

    Pipeline:
      1. Energy-based VAD → speech segments
      2. CAM++ xvec per segment → speaker embeddings
      3. Agglomerative clustering → speaker IDs
      4. F0 analysis → gender estimation

    Returns list of [Speaker, Gender, Age, Start(s), Duration(s)] rows.
    """
    import librosa
    from funcineforge.utils.load_utils import OnnxModel
    from sklearn.cluster import AgglomerativeClustering

    sr_vad = 16000
    wav, _ = librosa.load(audio_path, sr=sr_vad, mono=True)
    total_dur = len(wav) / sr_vad

    # ── Step 1: Energy-based VAD ──
    frame_len = int(0.025 * sr_vad)  # 25ms frames
    hop_len = int(0.010 * sr_vad)    # 10ms hop
    energy = np.array([
        np.sqrt(np.mean(wav[i:i + frame_len] ** 2))
        for i in range(0, len(wav) - frame_len, hop_len)
    ])
    # Adaptive threshold: max of fixed threshold and percentile
    threshold = max(energy_threshold, np.percentile(energy, 30))
    is_speech = energy > threshold

    # Convert frame-level decisions to segments
    raw_segments = []
    in_speech = False
    start_frame = 0
    for i, s in enumerate(is_speech):
        if s and not in_speech:
            start_frame = i
            in_speech = True
        elif not s and in_speech:
            start_t = start_frame * 0.010
            end_t = i * 0.010
            if end_t - start_t >= min_speech_dur:
                raw_segments.append((start_t, end_t))
            in_speech = False
    if in_speech:
        start_t = start_frame * 0.010
        end_t = len(is_speech) * 0.010
        if end_t - start_t >= min_speech_dur:
            raw_segments.append((start_t, end_t))

    if not raw_segments:
        # Fallback: single segment covering all
        return [["1", "男 Male", "中年 Adult", 0.0, round(total_dur, 2)]]

    # Merge segments with small gaps
    merged = [raw_segments[0]]
    for seg in raw_segments[1:]:
        gap = seg[0] - merged[-1][1]
        if gap < min_silence_dur:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    # ── Step 2: Extract CAM++ xvec per segment ──
    campp = OnnxModel(CAMPP_MODEL_PATH)
    embeddings = []
    for start_t, end_t in merged:
        s_idx = int(start_t * sr_vad)
        e_idx = int(end_t * sr_vad)
        seg_wav = wav[s_idx:e_idx]
        if len(seg_wav) < sr_vad * 0.1:  # skip < 100ms
            embeddings.append(np.zeros(192))
            continue
        xvec = campp(seg_wav)
        embeddings.append(xvec.flatten())

    embeddings = np.array(embeddings)

    # ── Step 3: Cluster speakers ──
    if len(merged) <= 1:
        labels = [0]
    else:
        n_clusters = min(max(2, len(set(range(len(merged))))), 6)
        # Estimate number of speakers: try 2-4, pick best silhouette
        from sklearn.metrics import silhouette_score
        best_score, best_labels = -1, None
        for n_c in range(2, min(len(merged), 5) + 1):
            try:
                clust = AgglomerativeClustering(
                    n_clusters=n_c, metric='cosine', linkage='average'
                )
                labs = clust.fit_predict(embeddings)
                if len(set(labs)) > 1:
                    score = silhouette_score(embeddings, labs, metric='cosine')
                    if score > best_score:
                        best_score = score
                        best_labels = labs
            except Exception:
                continue
        labels = best_labels if best_labels is not None else [0] * len(merged)

    # ── Step 4: Gender estimation via F0 ──
    gender_by_speaker = {}
    for i, (start_t, end_t) in enumerate(merged):
        spk = int(labels[i]) + 1
        if spk in gender_by_speaker:
            continue
        s_idx = int(start_t * sr_vad)
        e_idx = int(end_t * sr_vad)
        seg_wav = wav[s_idx:e_idx]
        try:
            f0, _, _ = librosa.pyin(
                seg_wav, fmin=60, fmax=500, sr=sr_vad,
                frame_length=2048
            )
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                median_f0 = np.median(f0_valid)
                gender_by_speaker[spk] = "女 Female" if median_f0 > 165 else "男 Male"
            else:
                gender_by_speaker[spk] = "男 Male"
        except Exception:
            gender_by_speaker[spk] = "男 Male"

    # ── Build result table ──
    result = []
    for i, (start_t, end_t) in enumerate(merged):
        spk = int(labels[i]) + 1
        dur = round(end_t - start_t, 2)
        gender = gender_by_speaker.get(spk, "男 Male")
        result.append([
            str(spk), gender, "中年 Adult",
            round(start_t, 2), dur,
        ])

    return result


# ──────────────────────────────────────────────
# Event handlers
# ──────────────────────────────────────────────
def on_scene_type_change(scene_type):
    """Toggle single-speaker vs multi-speaker panels."""
    is_dialogue = scene_type in DIALOGUE_TYPES
    return (
        gr.update(visible=not is_dialogue),   # single_speaker_col
        gr.update(visible=is_dialogue),        # dialogue_col
    )


def on_demo_select(evt: gr.SelectData):
    """Fill form fields when a demo item is selected."""
    items = load_demo_items()
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx < 0 or idx >= len(items):
        return [gr.update()] * 9

    item = items[idx]
    vocal = resolve_path(item["vocal"])
    video = resolve_path(item["video"])
    dlg = item["dialogue"]
    scene = TYPE_REV.get(item["type"], "独白 Monologue")
    is_dialogue = scene in DIALOGUE_TYPES

    # First speaker's gender/age for single-speaker panel
    first_gender = GENDER_REV.get(dlg[0]["gender"], "男 Male") if dlg else "男 Male"
    first_age = AGE_REV.get(dlg[0]["age"], "中年 Adult") if dlg else "中年 Adult"

    # Build dialogue table from demo data
    dlg_table = []
    for seg in dlg:
        dlg_table.append([
            str(seg.get("spk", "1")),
            GENDER_REV.get(seg["gender"], "男 Male"),
            AGE_REV.get(seg["age"], "中年 Adult"),
            float(seg.get("start", 0.0)),
            float(seg.get("duration", 3.0)),
        ])
    if not dlg_table:
        dlg_table = DEFAULT_DIALOGUE_TABLE

    return (
        item["text"],                                   # text
        item["clue"],                                   # clue
        scene,                                          # scene_type
        first_gender,                                   # gender (single mode)
        first_age,                                      # age (single mode)
        vocal if os.path.exists(vocal) else None,       # ref audio
        video if os.path.exists(video) else None,       # ref video
        f"demo:{item['utt']}",                         # demo_key
        dlg_table,                                      # dialogue_df
    )


def on_synthesize(
    text, clue, scene_type, gender, age,
    vocal_file, video_file, demo_key,
    dialogue_df,
    seed, min_len, max_len, sampling
):
    """Unified synthesis handler — single-speaker and multi-speaker.

    Demo mode: uses demo reference audio/video/face, but ALWAYS respects
    user's text and UI form settings.
    """
    if not text:
        raise gr.Error("⚠️ Please enter text to synthesize")

    is_dialogue = scene_type in DIALOGUE_TYPES

    # Build dialogue array from UI
    if is_dialogue and dialogue_df is not None and len(dialogue_df) > 0:
        dialogue = []
        for row in dialogue_df.values.tolist() if hasattr(dialogue_df, 'values') else dialogue_df:
            spk = str(row[0]) if row[0] else "1"
            g = GENDER_MAP.get(str(row[1]), str(row[1]))
            a = AGE_MAP.get(str(row[2]), str(row[2]))
            start = float(row[3]) if row[3] else 0.0
            dur = float(row[4]) if row[4] else 3.0
            dialogue.append({
                "start": start, "duration": dur,
                "spk": spk, "gender": g, "age": a,
            })
    else:
        # Single-speaker: one segment covering the whole duration
        est_dur = estimate_speech_length(text) / 25.0
        dialogue = [{
            "start": 0.0, "duration": round(est_dur, 2),
            "spk": "1",
            "gender": GENDER_MAP.get(gender, "male"),
            "age": AGE_MAP.get(age, "adult"),
        }]

    # Compute speech_length from dialogue timing in multi-speaker mode,
    # or from text length in single-speaker mode.
    # This is critical: speech_length MUST align with dialogue segment timing,
    # otherwise the model loses speaker assignment after the dialogue window.
    if is_dialogue and dialogue:
        max_end = max(seg["start"] + seg["duration"] for seg in dialogue)
        speech_length = max(50, int(max_end * 25))  # 25 fps
    else:
        speech_length = estimate_speech_length(text)

    # Demo mode: use demo's reference files
    is_demo = demo_key and demo_key.startswith("demo:")
    if is_demo:
        utt_name = demo_key.split(":", 1)[1]
        items = load_demo_items()
        item = next((it for it in items if it["utt"] == utt_name), None)
        if item is None:
            raise gr.Error(f"⚠️ Demo item '{utt_name}' not found")

        vocal_path = resolve_path(item["vocal"])
        video_path = resolve_path(item["video"])
        face_path = resolve_path(item["face"])
        utt_id = f"ref_{utt_name}"
    else:
        # Custom mode
        if vocal_file is None:
            raise gr.Error("⚠️ Please upload a reference audio file")

        vocal_path = vocal_file
        utt_id = f"custom_{int(time.time())}"
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
        vocal_path=vocal_path, video_path=video_path,
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


def on_auto_detect(ref_audio, ref_video):
    """Auto-detect dialogue segments from reference audio or video."""
    audio_path = None

    if ref_audio:
        audio_path = ref_audio
    elif ref_video:
        # Extract audio from video via ffmpeg
        import subprocess
        tmp_wav = os.path.join(OUTPUT_DIR, f"_tmp_extracted_{int(time.time())}.wav")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", ref_video,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                tmp_wav
            ], check=True, capture_output=True)
            audio_path = tmp_wav
        except Exception as e:
            raise gr.Error(f"⚠️ Failed to extract audio: {e}")
    else:
        raise gr.Error("⚠️ Please upload Reference Audio or Video first")

    try:
        dialogue_table = auto_detect_dialogue(audio_path)
    except Exception as e:
        raise gr.Error(f"⚠️ Auto-detection failed: {e}")
    finally:
        # Clean up extracted audio
        if ref_video and audio_path and audio_path != ref_audio:
            try:
                os.unlink(audio_path)
            except Exception:
                pass

    n_speakers = len(set(row[0] for row in dialogue_table))
    scene = "对话 Dialogue" if n_speakers == 2 else "多人 Multi-Speaker"
    return (
        scene,           # scene_type → auto switch to dialogue mode
        dialogue_table,  # dialogue_df
        f"✅ Detected {len(dialogue_table)} segments, {n_speakers} speakers",
    )


def on_clear_demo():
    """Clear the demo selection — reset to custom mode."""
    return (
        "",                     # text
        "",                     # clue
        "独白 Monologue",       # scene_type
        "男 Male",              # gender
        "中年 Adult",           # age
        None,                   # ref audio
        None,                   # ref video
        "",                     # demo_key
        DEFAULT_DIALOGUE_TABLE, # dialogue_df
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

        # Hidden state
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
                scene_type = gr.Dropdown(
                    list(TYPE_MAP.keys()), value="独白 Monologue",
                    label="Scene Type",
                )

                # ── Single-speaker controls (独白/旁白) ──
                with gr.Column(visible=True) as single_speaker_col:
                    with gr.Row():
                        gender = gr.Dropdown(
                            list(GENDER_MAP.keys()), value="男 Male",
                            label="Gender",
                        )
                        age = gr.Dropdown(
                            list(AGE_MAP.keys()), value="中年 Adult",
                            label="Age",
                        )

                # ── Multi-speaker controls (对话/多人) ──
                with gr.Column(visible=False) as dialogue_col:
                    with gr.Row():
                        gr.Markdown(
                            "### 🗣️ Dialogue Segments\n"
                            "*Each row = one speaker turn. Add/remove rows as needed.*"
                        )
                        detect_btn = gr.Button(
                            "🔍 Auto-Detect from Audio/Video",
                            size="sm", variant="secondary",
                        )
                    detect_status = gr.Markdown(value="", visible=True)
                    dialogue_df = gr.Dataframe(
                        headers=["Speaker", "Gender", "Age", "Start(s)", "Duration(s)"],
                        datatype=["str", "str", "str", "number", "number"],
                        value=DEFAULT_DIALOGUE_TABLE,
                        interactive=True,
                        column_count=(5, "fixed"),
                        row_count=(2, "dynamic"),
                        max_height=250,
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

        # ══════════════════════════════════════
        # Events
        # ══════════════════════════════════════

        # Scene type toggles single/multi speaker panels
        scene_type.change(
            on_scene_type_change,
            inputs=[scene_type],
            outputs=[single_speaker_col, dialogue_col],
        )

        # Demo select fills all form fields + dialogue table
        form_outputs = [text, clue, scene_type, gender, age,
                        ref_audio, ref_video, demo_key, dialogue_df]

        demo_table.select(on_demo_select, outputs=form_outputs)
        clear_btn.click(on_clear_demo, outputs=form_outputs)

        # Auto-detect dialogue from audio/video
        detect_btn.click(
            on_auto_detect,
            inputs=[ref_audio, ref_video],
            outputs=[scene_type, dialogue_df, detect_status],
        )

        # Synthesize
        synth_btn.click(
            on_synthesize,
            inputs=[text, clue, scene_type, gender, age,
                    ref_audio, ref_video, demo_key,
                    dialogue_df,
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
