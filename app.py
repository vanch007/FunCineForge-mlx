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
_BGM_PATH = None      # Background music track from vocal separation
_ORIG_SEGMENTS = []   # Original video timestamps for post-synthesis placement
_SEGMENT_TEXTS = []   # Per-segment translated text for chunked synthesis

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
    # Skip ref_ prefixed items (reference duplicates of originals)
    return [[i, it["utt"], it["type"],
             it["text"][:80] + ("..." if len(it["text"]) > 80 else "")]
            for i, it in enumerate(items) if not it["utt"].startswith("ref_")]


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

# Default dialogue table (2 speakers) — unified 7-column format
# Columns: Speaker | Gender | Age | Start(s) | Duration(s) | Text | Clue
DEFAULT_DIALOGUE_TABLE = [
    ["1", "男 Male", "中年 Adult", 0.0, 3.0, "Hello", "A male speaker greets warmly."],
    ["2", "女 Female", "中年 Adult", 3.0, 3.0, "Hi there", "A female speaker responds cheerfully."],
]


def generate_segment_clue(text, gender, age, spk_id, prev_text=None, next_text=None,
                          lang='zh', n_speakers=2, position='middle'):
    """Generate a context-aware clue for a dialogue segment.

    Analyzes the text content to determine emotional tone, speaking style,
    and generates a descriptive clue similar to the FunCineForge demo data.
    """
    gender_en = "male" if "Male" in gender else "female"
    gender_zh = "男性" if "Male" in gender else "女性"

    # Analyze text content for emotional cues
    is_question = '?' in text or '？' in text or '吗' in text or '呢' in text or '么' in text
    is_exclaim = '!' in text or '！' in text
    is_short = len(text) <= 4
    is_agreement = text.strip() in ('对', '嗯', '好', '是', '行', '没有', '没', '好的', '是的',
                                     'yes', 'yeah', 'ok', 'sure', 'right', 'hmm')
    has_emotion_words = any(w in text for w in ('哈', '呵', '嘿', '唉', '哎', '啊', '呀',
                                                 '天哪', '我靠', '他妈', '操'))
    is_angry = any(w in text for w in ('他妈', '操', '老子', '去你', '混蛋', '王八'))
    is_commanding = any(w in text for w in ('给我', '快', '马上', '立刻', '必须', '不准', '别'))
    is_mocking = any(w in text for w in ('还', '倒是', '你倒', '呵', '可笑'))
    is_assertive = any(w in text for w in ('就是', '当然', '一定', '肯定', '绝对'))
    is_worried = any(w in text for w in ('怕', '担心', '不行', '完了', '糟'))
    is_narrative = len(text) > 20 and not is_question and not is_exclaim

    # Determine position context
    is_first = position == 'first'
    is_responding = prev_text is not None

    if lang in ('zh', 'ja', 'ko'):
        # Chinese clue generation
        spk_desc = f"说话人{spk_id}，一位{gender_zh}"

        if is_agreement:
            tones = ["语气简短地表示认同", "以简洁的附和回应", "用简短的肯定作为回应"]
            import random
            tone = random.choice(tones)
        elif is_angry:
            tone = "语气愤怒而强烈，带有明显的不满和攻击性"
        elif is_commanding:
            tone = "以命令式的口吻说话，语气急促而有力"
        elif is_mocking:
            tone = "语气带有嘲讽和不屑，暗含挑衅"
        elif is_question and is_short:
            tone = "以简短的反问回应，语气中带有质疑"
        elif is_question:
            tone = "以疑问的语气提出问题，带有好奇或追问的意味"
        elif is_exclaim:
            tone = "语气激动，带有强烈的情绪"
        elif is_worried:
            tone = "语气中透露出担忧和不安"
        elif is_assertive:
            tone = "语气坚定而自信，态度明确"
        elif is_narrative and not is_responding:
            tone = "以陈述的语气叙述，声音沉稳"
        elif is_narrative and is_responding:
            tone = "接着话题继续阐述，语气沉稳"
        elif is_short and is_responding:
            tone = "以简短的语句回应，语气平淡"
        elif is_first:
            tone = "率先开口说话，语气自然"
        else:
            tone = "以自然的语气说话，情绪清晰"

        clue = f"{spk_desc}。{tone}。"
    else:
        # English clue generation
        spk_desc = f"Speaker {spk_id}, an adult {gender_en}"

        if is_agreement:
            tone = "gives a brief, affirming response"
        elif is_angry:
            tone = "speaks with anger and aggression, voice sharp and forceful"
        elif is_commanding:
            tone = "delivers a command with urgency and authority"
        elif is_mocking:
            tone = "speaks with a mocking and dismissive tone"
        elif is_question and is_short:
            tone = "asks a brief, pointed question with a skeptical tone"
        elif is_question:
            tone = "asks a question with curiosity or interrogation"
        elif is_exclaim:
            tone = "exclaims with heightened emotion"
        elif is_worried:
            tone = "speaks with worry and concern in voice"
        elif is_assertive:
            tone = "speaks with confidence and conviction"
        elif is_narrative and not is_responding:
            tone = "narrates steadily with a calm, descriptive tone"
        elif is_narrative and is_responding:
            tone = "continues the discussion with a measured tone"
        elif is_first:
            tone = "initiates the conversation with a natural tone"
        else:
            tone = "speaks with natural emotion and clear intonation"

        clue = f"{spk_desc}, {tone}."

    return clue


def translate_text(text, source='zh-CN', target='en'):
    """Translate text using free Google Translate API.

    Handles long text by splitting into chunks of max 4000 chars at sentence
    boundaries. Returns translated text or original on failure.
    """
    import urllib.request, urllib.parse

    MAX_CHUNK = 4000  # Google Translate API limit

    def _do_translate(chunk):
        url = ("https://translate.googleapis.com/translate_a/single"
               f"?client=gtx&sl={source}&tl={target}&dt=t"
               f"&q={urllib.parse.quote(chunk)}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            return ''.join([item[0] for item in data[0] if item[0]])
        except Exception as e:
            logger.warning(f"Translation failed for chunk: {e}")
            return chunk

    if len(text) <= MAX_CHUNK:
        return _do_translate(text)

    # Split at sentence boundaries for long texts
    import re
    sentences = re.split(r'([。！？.!?])', text)
    chunks = []
    current = ''
    for s in sentences:
        if len(current) + len(s) > MAX_CHUNK and current:
            chunks.append(current)
            current = s
        else:
            current += s
    if current:
        chunks.append(current)

    return ' '.join(_do_translate(c) for c in chunks)


def compact_dialogue_timeline(segments, gap=0.3):
    """Merge short adjacent segments, then compact into continuous timeline.

    The model expects segments of ~2-5s each (like demo data). This function:
      1. Merges adjacent SAME-SPEAKER segments with small gaps (< 0.5s)
      2. Caps merged segments at ~5s to match demo format
      3. Ensures no segment is shorter than ~1s (merges into same-speaker neighbor)
      4. Compacts the timeline by removing large gaps

    Args:
        segments: list of [spk, gender, age, start, duration] from auto-detect
        gap: small gap between segments in seconds for compacted output

    Returns:
        (compacted_segments, merge_map) where merge_map[i] = list of original
        segment indices that were merged into compacted segment i.
    """
    if not segments:
        return segments, []

    MAX_SEG_DUR = 5.0   # Max duration per segment (demo avg ~2-5s)
    MERGE_GAP = 0.5     # Only merge if gap is < this
    MIN_SEG_DUR = 1.0   # Minimum useful segment duration

    # Sort by original start time, keep track of original indices
    indexed_segs = sorted(enumerate(segments), key=lambda x: float(x[1][3]))

    # Step 1: Merge adjacent same-speaker segments with short gaps, cap at MAX_SEG_DUR
    merged = [list(indexed_segs[0][1])]
    merge_groups = [[indexed_segs[0][0]]]  # track which orig indices
    for orig_idx, seg in indexed_segs[1:]:
        prev = merged[-1]
        prev_end = float(prev[3]) + float(prev[4])
        cur_start = float(seg[3])
        gap_between = cur_start - prev_end
        same_speaker = str(prev[0]) == str(seg[0])
        merged_dur = (cur_start + float(seg[4])) - float(prev[3])

        # Merge only if: same speaker + short gap + won't exceed max duration
        if same_speaker and gap_between < MERGE_GAP and merged_dur <= MAX_SEG_DUR:
            prev[4] = round(merged_dur, 3)
            merge_groups[-1].append(orig_idx)
        else:
            merged.append(list(seg))
            merge_groups.append([orig_idx])

    # Step 2: Merge sub-1s segments into SAME-SPEAKER neighbor only
    final = []
    final_groups = []
    for i, seg in enumerate(merged):
        dur = float(seg[4])
        if dur < MIN_SEG_DUR and final:
            prev = final[-1]
            same_speaker = str(prev[0]) == str(seg[0])
            combined = (float(seg[3]) + dur) - float(prev[3])
            if same_speaker and combined <= MAX_SEG_DUR:
                prev[4] = round(combined, 3)
                final_groups[-1].extend(merge_groups[i])
                continue
        final.append(list(seg))
        final_groups.append(list(merge_groups[i]))

    # If still have tiny segments at the start, only merge if same speaker
    if final and float(final[0][4]) < MIN_SEG_DUR and len(final) > 1:
        if str(final[0][0]) == str(final[1][0]):
            final[1][3] = final[0][3]
            final[1][4] = round(float(final[1][4]) + float(final[0][4]) + 0.3, 3)
            final_groups[1] = final_groups[0] + final_groups[1]
            final.pop(0)
            final_groups.pop(0)

    if not final:
        all_orig = [idx for g in merge_groups for idx in g]
        final = [list(indexed_segs[0][1])]
        final[0][4] = round(max(float(indexed_segs[-1][1][3]) + float(indexed_segs[-1][1][4]) - float(indexed_segs[0][1][3]), 1.0), 3)
        final_groups = [all_orig]

    # Step 3: Compact into continuous timeline (remove large gaps)
    compacted = []
    cursor = 0.0
    for seg in final:
        dur = float(seg[4])
        compacted.append([seg[0], seg[1], seg[2], round(cursor, 3), round(dur, 3)])
        cursor += dur + gap

    return compacted, final_groups

CAMPP_MODEL_PATH = os.path.join(CKPT_DIR, "camplus.onnx")


def auto_detect_dialogue(audio_path, min_speech_dur=0.2, min_silence_dur=0.15,
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

    # ── Step 3: Cluster speakers using spectral clustering ──
    # SpectralClustering on cosine affinity works better than agglomerative
    # for CAM++ embeddings which have relatively high intra-speaker similarity.
    from sklearn.preprocessing import normalize
    from sklearn.cluster import SpectralClustering

    if len(merged) <= 1:
        labels = [0]
    else:
        emb_norm = normalize(embeddings, norm='l2')
        # Build cosine affinity matrix (shifted to [0,1])
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(emb_norm)
        affinity = (sim + 1) / 2  # cosine sim [-1,1] → [0,1]
        np.fill_diagonal(affinity, 0)

        # Estimate number of speakers: try 2-4, pick best silhouette
        from sklearn.metrics import silhouette_score
        best_score, best_labels = -1, None
        for n_c in range(2, min(len(merged), 5) + 1):
            try:
                clust = SpectralClustering(
                    n_clusters=n_c, affinity='precomputed', random_state=42
                )
                labs = clust.fit_predict(affinity)
                if len(set(labs)) > 1:
                    score = silhouette_score(emb_norm, labs, metric='cosine')
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
    full_text = item["text"]
    full_clue = item["clue"]

    # First speaker's gender/age for single-speaker panel
    first_gender = GENDER_REV.get(dlg[0]["gender"], "男 Male") if dlg else "男 Male"
    first_age = AGE_REV.get(dlg[0]["age"], "中年 Adult") if dlg else "中年 Adult"

    # Split text into sentences, then distribute among segments
    import re
    sentences = re.split(r'(?<=[.!?。！？])\s*', full_text.strip())
    sentences = [s for s in sentences if s.strip()]

    n_seg = len(dlg)
    if n_seg == 1:
        seg_texts = [full_text.strip()]
    elif len(sentences) == n_seg:
        seg_texts = sentences
    elif len(sentences) > n_seg:
        # More sentences than segments: distribute proportionally by duration
        total_dur = sum(s.get("duration", 1.0) for s in dlg)
        seg_texts = []
        sent_idx = 0
        for si, seg in enumerate(dlg):
            frac = seg.get("duration", 1.0) / total_dur if total_dur > 0 else 1.0 / n_seg
            n_sents = max(1, round(frac * len(sentences)))
            if si == n_seg - 1:
                # Last segment gets all remaining
                seg_texts.append(' '.join(sentences[sent_idx:]))
            else:
                end = min(sent_idx + n_sents, len(sentences))
                seg_texts.append(' '.join(sentences[sent_idx:end]))
                sent_idx = end
    else:
        # Fewer sentences than segments: first segments get one each, rest empty
        seg_texts = sentences + [''] * (n_seg - len(sentences))

    # Build unified table (7 columns) from demo data
    dlg_table = []
    for i, seg in enumerate(dlg):
        seg_text = seg_texts[i] if i < len(seg_texts) else ''
        dlg_table.append([
            str(seg.get("spk", "1")),
            GENDER_REV.get(seg["gender"], "男 Male"),
            AGE_REV.get(seg["age"], "中年 Adult"),
            float(seg.get("start", 0.0)),
            float(seg.get("duration", 3.0)),
            seg_text,
            full_clue,  # Global clue for each segment (user can edit per-segment)
        ])
    if not dlg_table:
        dlg_table = DEFAULT_DIALOGUE_TABLE

    return (
        full_text,                                      # text
        full_clue,                                      # clue
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

    The dialogue_df now has 7 columns: Speaker|Gender|Age|Start|Duration|Text|Clue.
    Per-segment text/clue are extracted from columns 5/6.
    """
    global _SEGMENT_TEXTS

    if not text:
        raise gr.Error("⚠️ Please enter text to synthesize")

    is_dialogue = scene_type in DIALOGUE_TYPES

    # Build dialogue array from unified 7-column table
    if is_dialogue and dialogue_df is not None and len(dialogue_df) > 0:
        dialogue = []
        seg_texts_from_table = []
        seg_clues_from_table = []
        for row in dialogue_df.values.tolist() if hasattr(dialogue_df, 'values') else dialogue_df:
            spk = str(row[0]) if row[0] else "1"
            g = GENDER_MAP.get(str(row[1]), str(row[1]))
            a = AGE_MAP.get(str(row[2]), str(row[2]))
            start = float(row[3]) if row[3] else 0.0
            dur = float(row[4]) if row[4] else 3.0
            seg_text = str(row[5]) if len(row) > 5 and row[5] else ""
            seg_clue = str(row[6]) if len(row) > 6 and row[6] else ""
            dialogue.append({
                "start": start, "duration": dur,
                "spk": spk, "gender": g, "age": a,
            })
            seg_texts_from_table.append(seg_text)
            seg_clues_from_table.append(seg_clue)
        # Update _SEGMENT_TEXTS from the table (user may have edited)
        _SEGMENT_TEXTS = seg_texts_from_table
        # If text field is empty/generic, rebuild from table texts
        if not text or text.startswith("[Speaker"):
            text = ' '.join(t for t in seg_texts_from_table if t)
    else:
        # Single-speaker: one segment covering the whole duration
        est_dur = estimate_speech_length(text) / 25.0
        dialogue = [{
            "start": 0.0, "duration": round(est_dur, 2),
            "spk": "1",
            "gender": GENDER_MAP.get(gender, "male"),
            "age": AGE_MAP.get(age, "adult"),
        }]
        seg_clues_from_table = [clue] if clue else []

    # Compute speech_length from dialogue timing in multi-speaker mode,
    # or from text length in single-speaker mode.
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

    MAX_CHUNK_FRAMES = 375  # ~15s: model works best within this range

    t0 = time.time()

    if speech_length > MAX_CHUNK_FRAMES and is_dialogue and len(dialogue) > 1:
        # Chunked synthesis for long videos
        import soundfile as sf
        logger.info(f"speech_length={speech_length} exceeds {MAX_CHUNK_FRAMES}, using chunked synthesis")

        # Split dialogue into chunks of max ~15s, tracking segment indices
        chunks = []       # list of (seg_list, seg_index_list)
        cur_chunk = []
        cur_indices = []
        chunk_base = 0.0
        for si, seg in enumerate(dialogue):
            seg_end = seg["start"] + seg["duration"]
            if seg_end - chunk_base > MAX_CHUNK_FRAMES / 25.0 and cur_chunk:
                chunks.append((cur_chunk, cur_indices))
                cur_chunk = [seg]
                cur_indices = [si]
                chunk_base = seg["start"]
            else:
                cur_chunk.append(seg)
                cur_indices.append(si)
        if cur_chunk:
            chunks.append((cur_chunk, cur_indices))

        chunk_wavs = []
        vid_path = None
        for ci, (chunk, chunk_seg_indices) in enumerate(chunks):
            # Make chunk dialogue relative to chunk start
            cbase = chunk[0]["start"]
            chunk_dlg = [
                {**s, "start": s["start"] - cbase}
                for s in chunk
            ]
            c_end = max(d["start"] + d["duration"] for d in chunk_dlg)
            c_sl = max(50, int(c_end * 25))

            # Per-chunk text: use segment-level translations if available
            if _SEGMENT_TEXTS and len(_SEGMENT_TEXTS) == len(dialogue):
                chunk_text = ' '.join(_SEGMENT_TEXTS[i] for i in chunk_seg_indices if i < len(_SEGMENT_TEXTS))
            else:
                chunk_text = text  # fallback to full text

            # Per-chunk clue: combine per-segment clues from the unified table
            chunk_seg_clues = [seg_clues_from_table[i] for i in chunk_seg_indices
                               if i < len(seg_clues_from_table) and seg_clues_from_table[i]]
            if chunk_seg_clues:
                # Use the per-segment clues from the table (user-editable)
                chunk_clue = ' '.join(chunk_seg_clues)
            else:
                # Fallback: auto-generate from speaker metadata
                chunk_spk_info = {}
                for s in chunk_dlg:
                    spk_id = s.get("spk", "1")
                    if spk_id not in chunk_spk_info:
                        chunk_spk_info[spk_id] = {
                            "gender": s.get("gender", "male"),
                            "age": s.get("age", "adult"),
                        }
                spk_descs = [f"Speaker {sid} is an {info['age']} {info['gender']}"
                             for sid, info in sorted(chunk_spk_info.items())]
                chunk_clue = f"{'. '.join(spk_descs)}. Natural emotion and clear intonation."

            c_face = create_empty_face_pkl(c_sl) if not is_demo else face_path
            c_data = build_jsonl_item(
                utt=f"{utt_id}_c{ci}", text=chunk_text,
                clue=chunk_clue,
                scene_type=TYPE_MAP.get(scene_type, "独白"),
                vocal_path=vocal_path, video_path=video_path,
                face_path=c_face, dialogue=chunk_dlg, speech_length=c_sl,
            )
            c_wav, c_vid = run_inference(
                c_data, seed=int(seed), min_len=int(min_len),
                max_len=int(max_len), sampling=sampling
            )
            if c_wav:
                chunk_wavs.append(c_wav)
            if c_vid and not vid_path:
                vid_path = c_vid
            if not is_demo and c_face != face_path:
                try: os.unlink(c_face)
                except: pass
            logger.info(f"Chunk {ci+1}/{len(chunks)} done")

        # Concatenate chunks
        if chunk_wavs:
            all_audio = []
            sr_out = 24000
            for w in chunk_wavs:
                audio_data, sr_out = sf.read(w)
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                all_audio.append(audio_data)
            import numpy as np
            combined = np.concatenate(all_audio)
            wav_path = os.path.join(OUTPUT_DIR, "wav", f"{utt_id}_combined.wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            sf.write(wav_path, combined, sr_out, subtype='PCM_16')
            logger.info(f"Combined {len(chunks)} chunks: {len(combined)/sr_out:.1f}s")
        else:
            wav_path = None
    else:
        # Single-shot synthesis (short content)
        data = build_jsonl_item(
            utt=utt_id, text=text,
            clue=clue or "A speaker speaks clearly with natural emotion.",
            scene_type=TYPE_MAP.get(scene_type, "独白"),
            vocal_path=vocal_path, video_path=video_path,
            face_path=face_path, dialogue=dialogue,
            speech_length=speech_length,
        )
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

    # Place generated audio on original video timeline (if we have original segment info)
    if _ORIG_SEGMENTS and _BGM_PATH and wav_path and os.path.exists(wav_path):
        try:
            placed_path = place_audio_on_timeline(
                wav_path, _ORIG_SEGMENTS, _BGM_PATH
            )
            logger.info(f"Audio placed on timeline + BGM mixed: {placed_path}")
            wav_path = placed_path
        except Exception as e:
            logger.warning(f"Timeline placement failed, trying simple mix: {e}")
            # Fallback: simple BGM mix without timeline placement
            try:
                mixed_path = mix_audio_with_bgm(wav_path, _BGM_PATH)
                wav_path = mixed_path
            except Exception:
                pass
    elif _BGM_PATH and wav_path and os.path.exists(wav_path):
        try:
            mixed_path = mix_audio_with_bgm(wav_path, _BGM_PATH)
            logger.info(f"Mixed BGM into output: {mixed_path}")
            wav_path = mixed_path
        except Exception as e:
            logger.warning(f"BGM mixing failed, using vocals only: {e}")

    # Re-merge mixed audio into video (if video was generated)
    if vid_path and os.path.exists(vid_path) and wav_path:
        import subprocess
        mixed_vid = vid_path.replace(".mp4", "_final.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", vid_path, "-i", wav_path,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", mixed_vid
            ], check=True, capture_output=True)
            vid_path = mixed_vid
        except Exception as e:
            logger.warning(f"Video re-mux failed: {e}")

    return wav_path, vid_path, f"✅ Done in {elapsed:.1f}s"


def _extract_audio_from_source(ref_audio, ref_video):
    """Get audio path from ref_audio or extract from ref_video."""
    if ref_audio:
        return ref_audio, False  # path, is_temp
    if ref_video:
        import subprocess
        tmp_wav = os.path.join(OUTPUT_DIR, f"_tmp_extracted_{int(time.time())}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", ref_video,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_wav
        ], check=True, capture_output=True)
        return tmp_wav, True
    return None, False


def separate_vocals(audio_path, output_dir=None):
    """Separate vocals from background using demucs htdemucs.

    Returns (vocals_path, bgm_path).
    """
    import subprocess
    out_dir = output_dir or OUTPUT_DIR
    sep_dir = os.path.join(out_dir, "_separated")
    os.makedirs(sep_dir, exist_ok=True)

    python_bin = sys.executable
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    result = subprocess.run([
        python_bin, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-d", device,
        "-o", sep_dir,
        audio_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-300:]}")

    # Find output files — demucs outputs to sep_dir/htdemucs/<basename>/
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(sep_dir, "htdemucs", basename)
    vocals_path = os.path.join(stem_dir, "vocals.wav")
    bgm_path = os.path.join(stem_dir, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Vocals not found at {vocals_path}")

    return vocals_path, bgm_path


def mix_audio_with_bgm(generated_wav_path, bgm_path, output_path=None):
    """Mix generated vocals with background music track."""
    import subprocess
    if not bgm_path or not os.path.exists(bgm_path):
        return generated_wav_path
    if not output_path:
        output_path = generated_wav_path.replace(".wav", "_mixed.wav")

    # Mix: generated vocals + BGM, trim to shorter
    subprocess.run([
        "ffmpeg", "-y",
        "-i", generated_wav_path,
        "-i", bgm_path,
        "-filter_complex",
        "[0:a]apad[a];[a][1:a]amix=inputs=2:duration=shortest:weights=1 0.5[out]",
        "-map", "[out]",
        "-ac", "1", "-ar", "24000",
        output_path
    ], check=True, capture_output=True)

    return output_path



def place_audio_on_timeline(generated_wav, orig_segments, bgm_path):
    """Place compact TTS audio segments at original video timeline positions.

    The TTS model generates one continuous audio for all dialogue segments.
    This function splits it back and places each piece at the original position,
    with BGM filling the gaps.

    Args:
        generated_wav: path to TTS output (compact audio)
        orig_segments: list of dicts with original {start, duration, ...}
        bgm_path: background music track

    Returns:
        output_path: mixed audio at original timeline positions
    """
    import subprocess
    import soundfile as sf
    import numpy as np
    import librosa

    # Read the generated audio and BGM
    gen_audio, gen_sr = sf.read(generated_wav)
    bgm_audio, bgm_sr = sf.read(bgm_path)

    # Convert to mono if stereo
    if gen_audio.ndim > 1:
        gen_audio = gen_audio.mean(axis=1)
    if bgm_audio.ndim > 1:
        bgm_audio = bgm_audio.mean(axis=1)

    # Resample BGM to match generated audio sample rate if needed
    if bgm_sr != gen_sr:
        bgm_audio = librosa.resample(bgm_audio, orig_sr=bgm_sr, target_sr=gen_sr)

    # Output length = length of BGM (which matches original video)
    out_len = len(bgm_audio)
    output = np.zeros(out_len, dtype=np.float32)

    # Copy BGM as base at 50% volume
    output[:out_len] = bgm_audio[:out_len] * 0.5

    # Split generated audio by compact segment durations and place at original positions
    gen_cursor = 0
    for seg in sorted(orig_segments, key=lambda s: s["start"]):
        orig_start_sample = int(seg["start"] * gen_sr)
        dur_samples = int(seg["duration"] * gen_sr)
        # Add a small gap between compact segments (0.3s gap was used in compaction)
        compact_dur_with_gap = dur_samples + int(0.3 * gen_sr)

        # Extract the segment from generated audio
        gen_end = min(gen_cursor + dur_samples, len(gen_audio))
        seg_audio = gen_audio[gen_cursor:gen_end]

        # Place at original position in output
        place_end = min(orig_start_sample + len(seg_audio), out_len)
        place_len = place_end - orig_start_sample
        if place_len > 0 and orig_start_sample >= 0:
            output[orig_start_sample:place_end] = (
                output[orig_start_sample:place_end] * 0.3 +  # reduce BGM under speech
                seg_audio[:place_len]
            )

        gen_cursor += compact_dur_with_gap  # skip the gap used in compaction

    # Save output
    output_path = generated_wav.replace(".wav", "_placed.wav")
    sf.write(output_path, output, gen_sr, subtype='PCM_16')
    return output_path



def on_auto_analyze(ref_audio, ref_video, progress=gr.Progress()):
    """Full auto-analysis: vocal separation + ASR + VAD + speaker diarization.

    Fills: text, clue, scene_type, gender, age, dialogue_df, detect_status, ref_audio.
    """
    global _BGM_PATH, _ORIG_SEGMENTS, _SEGMENT_TEXTS

    try:
        audio_path, is_temp = _extract_audio_from_source(ref_audio, ref_video)
    except Exception as e:
        raise gr.Error(f"⚠️ Failed to extract audio: {e}")
    if not audio_path:
        raise gr.Error("⚠️ Please upload Reference Audio or Video first")

    vocals_path = audio_path  # fallback if separation fails
    try:
        # ── Step 1: Vocal/BGM separation ──
        progress(0.05, desc="Separating vocals from background...")
        try:
            vocals_path, bgm_path = separate_vocals(audio_path)
            _BGM_PATH = bgm_path
            logger.info(f"Vocal separation done: vocals={vocals_path}, bgm={bgm_path}")
        except Exception as e:
            logger.warning(f"Vocal separation failed, using original: {e}")
            _BGM_PATH = None
            vocals_path = audio_path

        # ── Step 2: ASR on vocals only (with segment timestamps) ──
        progress(0.3, desc="Transcribing vocals...")
        try:
            import mlx_whisper
            asr_result = mlx_whisper.transcribe(
                vocals_path,
                path_or_hf_repo='mlx-community/whisper-large-v3-turbo',
                language=None,
            )
            logger.info("ASR: using mlx-whisper large-v3-turbo (MLX-native)")
        except ImportError:
            import whisper
            asr_model = whisper.load_model("base")
            asr_result = asr_model.transcribe(vocals_path, language=None)
            logger.info("ASR: using whisper base (fallback)")
        text = asr_result["text"].strip()
        lang = asr_result.get("language", "unknown")
        asr_segments = asr_result.get("segments", [])
        logger.info(f"ASR result: lang={lang}, {len(asr_segments)} segments, text={text[:100]}")

        # ── Step 3: ASR-segment-based speaker diarization ──
        # Use ASR sentence boundaries (more accurate than energy VAD) for speaker embedding.
        # Each ASR segment gets its own CAM++ xvec → SpectralClustering → speaker label.
        progress(0.6, desc="Detecting speakers...")

        import librosa
        from funcineforge.utils.load_utils import OnnxModel
        from sklearn.preprocessing import normalize
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics import silhouette_score

        sr_vad = 16000
        wav, _ = librosa.load(vocals_path, sr=sr_vad, mono=True)

        # Extract CAM++ xvec per ASR segment
        campp = OnnxModel(CAMPP_MODEL_PATH)
        asr_times = []  # (start, end, text) per valid ASR segment
        embeddings = []
        for aseg in asr_segments:
            a_start = aseg.get('start', 0)
            a_end = aseg.get('end', a_start)
            a_text = aseg.get('text', '').strip()
            if a_end - a_start < 0.15 or not a_text:
                continue
            s_idx = int(a_start * sr_vad)
            e_idx = int(a_end * sr_vad)
            seg_wav = wav[s_idx:e_idx]
            if len(seg_wav) < sr_vad * 0.15:
                continue
            xvec = campp(seg_wav)
            embeddings.append(xvec.flatten())
            asr_times.append((a_start, a_end, a_text))

        if not asr_times:
            # Fallback: single segment
            asr_times = [(0, len(wav)/sr_vad, text)]
            embeddings = [np.zeros(192)]

        embeddings = np.array(embeddings)

        # Cluster speakers
        if len(asr_times) <= 1:
            labels = [0]
        else:
            emb_norm = normalize(embeddings, norm='l2')
            sim = cosine_similarity(emb_norm)
            affinity = (sim + 1) / 2
            np.fill_diagonal(affinity, 0)

            best_score, best_labels = -1, None
            for n_c in range(2, min(len(asr_times), 5) + 1):
                try:
                    clust = SpectralClustering(
                        n_clusters=n_c, affinity='precomputed', random_state=42
                    )
                    labs = clust.fit_predict(affinity)
                    if len(set(labs)) > 1:
                        score = silhouette_score(emb_norm, labs, metric='cosine')
                        if score > best_score:
                            best_score = score
                            best_labels = labs
                except Exception:
                    continue
            labels = best_labels if best_labels is not None else [0] * len(asr_times)

        # Gender estimation via F0
        gender_by_speaker = {}
        for i, (a_start, a_end, _) in enumerate(asr_times):
            spk = int(labels[i]) + 1
            if spk in gender_by_speaker:
                continue
            seg_wav = wav[int(a_start * sr_vad):int(a_end * sr_vad)]
            try:
                f0, _, _ = librosa.pyin(seg_wav, fmin=60, fmax=500, sr=sr_vad, frame_length=2048)
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 0:
                    gender_by_speaker[spk] = "女 Female" if np.median(f0_valid) > 165 else "男 Male"
                else:
                    gender_by_speaker[spk] = "男 Male"
            except Exception:
                gender_by_speaker[spk] = "男 Male"

        # Build dialogue table and segment texts from ASR segments directly
        dialogue_table = []
        _SEGMENT_TEXTS = []
        for i, (a_start, a_end, a_text) in enumerate(asr_times):
            spk = int(labels[i]) + 1
            gender = gender_by_speaker.get(spk, "男 Male")
            dialogue_table.append([
                str(spk), gender, "中年 Adult",
                round(a_start, 2), round(a_end - a_start, 2),
            ])
            _SEGMENT_TEXTS.append(a_text)

        logger.info(f"ASR-based diarization: {len(asr_times)} segments, "
                     f"{len(set(labels))} speakers")

        # ── Step 4: Compact timeline + derive scene type ──
        progress(0.8, desc="Building compact timeline...")
        n_speakers = len(set(int(labels[i]) + 1 for i in range(len(labels))))
        n_segments = len(dialogue_table)

        # Save original timestamps before compaction
        orig_timestamps = [(float(row[3]), float(row[3]) + float(row[4])) for row in dialogue_table]

        # Compact segments into continuous timeline
        dialogue_table, merge_map = compact_dialogue_timeline(dialogue_table)
        n_compacted = len(dialogue_table)

        # Build _ORIG_SEGMENTS from compacted segments mapped back to original time ranges.
        # Each compacted segment covers the time span of all merged original segments.
        # This ensures place_audio_on_timeline gets exactly n_compacted entries.
        _ORIG_SEGMENTS = []
        for ci, row in enumerate(dialogue_table):
            if ci < len(merge_map) and merge_map[ci]:
                ois = merge_map[ci]
                orig_start = min(orig_timestamps[oi][0] for oi in ois if oi < len(orig_timestamps))
                orig_end = max(orig_timestamps[oi][1] for oi in ois if oi < len(orig_timestamps))
            else:
                orig_start = float(row[3])
                orig_end = orig_start + float(row[4])
            _ORIG_SEGMENTS.append({
                "start": orig_start,
                "duration": float(row[4]),  # compacted duration (matches TTS output length)
                "spk": str(row[0]),
                "gender": "male" if "Male" in row[1] else "female",
                "age": "adult",
            })

        # Rebuild _SEGMENT_TEXTS after compaction using merge_map
        orig_seg_texts = list(_SEGMENT_TEXTS)  # save pre-compact texts
        _SEGMENT_TEXTS = []
        for ci in range(n_compacted):
            if ci < len(merge_map) and merge_map[ci]:
                merged_text = ' '.join(orig_seg_texts[oi] for oi in merge_map[ci]
                                       if oi < len(orig_seg_texts) and orig_seg_texts[oi])
            else:
                merged_text = ''
            _SEGMENT_TEXTS.append(merged_text)

        logger.info(f"Compacted: {n_segments} → {n_compacted} segments, "
                     f"_ORIG_SEGMENTS: {len(_ORIG_SEGMENTS)} entries")

        # ── Step 5: Generate per-segment clues ──
        progress(0.9, desc="Generating per-segment descriptions...")

        lang_name = {"zh": "Chinese", "en": "English", "ja": "Japanese",
                     "ko": "Korean"}.get(lang, lang)

        # Build per-segment clue using context-aware generator
        seg_clues = []
        for i, row in enumerate(dialogue_table):
            seg_text = _SEGMENT_TEXTS[i] if i < len(_SEGMENT_TEXTS) else ''
            prev_text = _SEGMENT_TEXTS[i-1] if i > 0 and i-1 < len(_SEGMENT_TEXTS) else None
            next_text = _SEGMENT_TEXTS[i+1] if i+1 < len(_SEGMENT_TEXTS) else None
            position = 'first' if i == 0 else ('last' if i == len(dialogue_table) - 1 else 'middle')
            seg_clue = generate_segment_clue(
                text=seg_text, gender=row[1], age=row[2], spk_id=row[0],
                prev_text=prev_text, next_text=next_text,
                lang=lang, n_speakers=n_speakers, position=position,
            )
            seg_clues.append(seg_clue)

        # Build unified 7-column table: Speaker|Gender|Age|Start|Duration|Text|Clue
        unified_table = []
        for i, row in enumerate(dialogue_table):
            seg_text = _SEGMENT_TEXTS[i] if i < len(_SEGMENT_TEXTS) else ''
            seg_clue = seg_clues[i] if i < len(seg_clues) else ''
            unified_table.append([
                row[0], row[1], row[2],
                row[3], row[4],
                seg_text, seg_clue,
            ])

        # Text field: join all segment texts for display/fallback
        text_lines = []
        for i, row in enumerate(unified_table):
            seg_text = row[5]
            if seg_text:
                text_lines.append(f"[Speaker {row[0]}] {seg_text}")
        text = '\n'.join(text_lines) if text_lines else asr_result['text'].strip()

        if n_speakers == 1:
            scene = "独白 Monologue"
        elif n_speakers == 2:
            scene = "对话 Dialogue"
        else:
            scene = "多人 Multi-Speaker"

        first_gender = dialogue_table[0][1] if dialogue_table else "男 Male"
        first_age = dialogue_table[0][2] if dialogue_table else "中年 Adult"

        # Global clue (for single-speaker fallback and text field display)
        clue = (f"A {lang_name} conversation. "
                f"The speakers talk with natural emotion and clear intonation.")

        bgm_note = " | 🎵 BGM separated" if _BGM_PATH else ""
        status = f"✅ ASR ({lang_name}): {len(text)} chars | " + \
                 f"{n_compacted} segments, {n_speakers} speakers{bgm_note}"

    except Exception as e:
        raise gr.Error(f"⚠️ Auto-analysis failed: {e}")
    finally:
        if is_temp and audio_path:
            try:
                os.unlink(audio_path)
            except Exception:
                pass

    return (
        text,            # text
        clue,            # clue
        scene,           # scene_type
        first_gender,    # gender (single-speaker)
        first_age,       # age (single-speaker)
        unified_table,   # dialogue_df (now 7 columns)
        status,          # detect_status
        vocals_path,     # ref_audio → replace with vocals-only
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
            # Left column: Video + Audio + Analyze
            with gr.Column(scale=1):
                ref_video = gr.Video(label="🎬 Video to Dub (optional)")
                ref_audio = gr.Audio(
                    label="🎤 Reference Audio (voice timbre)", type="filepath",
                )
                analyze_btn = gr.Button(
                    "🔍 Auto-Analyze Audio/Video",
                    variant="secondary", size="sm",
                )
                detect_status = gr.Markdown(value="")

            # Right column: Scene controls + Dialogue table
            with gr.Column(scale=2):
                # Hidden state for text/clue (used by synthesizer)
                text = gr.Textbox(visible=False, value="")
                clue = gr.Textbox(visible=False, value="")

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
                    gr.Markdown(
                        "### 🗣️ Dialogue Segments\n"
                        "*Each row = one speaker turn with its text and clue. Editable.*"
                    )
                    dialogue_df = gr.Dataframe(
                        headers=["Speaker", "Gender", "Age", "Start(s)", "Duration(s)", "Text", "Clue"],
                        datatype=["str", "str", "str", "number", "number", "str", "str"],
                        value=DEFAULT_DIALOGUE_TABLE,
                        interactive=True,
                        column_count=(7, "fixed"),
                        row_count=(2, "dynamic"),
                        max_height=400,
                        wrap=True,
                    )

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

        # Auto-analyze: ASR + VAD + speaker diarization + vocal separation
        analyze_btn.click(
            on_auto_analyze,
            inputs=[ref_audio, ref_video],
            outputs=[text, clue, scene_type, gender, age,
                     dialogue_df, detect_status, ref_audio],
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
