#!/usr/bin/env python3
"""FunCineForge MPS Inference Benchmark — Captures per-item timing for each stage"""
import faulthandler
faulthandler.enable()

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('/Users/vanch/FunCineForge/exps')

import torch
import logging
import re
from funcineforge import AutoModel
from funcineforge.utils.hinter import get_logger
from funcineforge.models.utils import dtype_map
from funcineforge.register import tables
import hydra
from omegaconf import DictConfig

# ── Monkey-patch inference_model to capture per-stage timing ──
import funcineforge.models.inference_model as im

_orig_inference = im.FunCineForgeInferModel.inference
_timing_data = []

@torch.no_grad()
def _timed_inference(self, data_in, data_lengths=None, key=None, **kwargs):
    uttid = key[0]
    t0 = time.perf_counter()

    # LLM stage
    kwargs["tokenizer"] = self.tokenizer
    from funcineforge.utils.set_all_random_seed import set_all_random_seed
    set_all_random_seed(kwargs.get("random_seed", 0))
    t_llm_start = time.perf_counter()
    codec, hit_eos, states = self.lm_model.inference(data_in, data_lengths, key, **kwargs)
    t_llm_end = time.perf_counter()
    llm_ms = (t_llm_end - t_llm_start) * 1000
    gen_len = codec.shape[1]

    wav, batch_data_time = None, 1.0
    fm_ms, voc_ms = 0.0, 0.0

    if gen_len > 0:
        # FM stage
        data_in[0]["codec"] = codec
        set_all_random_seed(kwargs.get("random_seed", 0))
        t_fm_start = time.perf_counter()
        feat = self.fm_model.inference(data_in, data_lengths, key, **kwargs)
        t_fm_end = time.perf_counter()
        fm_ms = (t_fm_end - t_fm_start) * 1000

        # VOC stage
        set_all_random_seed(kwargs.get("random_seed", 0))
        t_voc_start = time.perf_counter()
        wav = self.voc_model.inference([feat[0]], data_lengths, key, **kwargs)
        t_voc_end = time.perf_counter()
        voc_ms = (t_voc_end - t_voc_start) * 1000

        # Save wav
        import soundfile as sf
        output_dir = kwargs.get("output_dir", None)
        if output_dir:
            wav_out_dir = os.path.join(output_dir, "wav")
            os.makedirs(wav_out_dir, exist_ok=True)
            sf.write(
                os.path.join(wav_out_dir, f"{key[0]}.wav"),
                wav.cpu().numpy().T,
                samplerate=self.sample_rate,
                subtype='PCM_16'
            )

        batch_data_time = wav.shape[1] / self.sample_rate

    total_ms = (time.perf_counter() - t0) * 1000
    rtf = total_ms / 1000.0 / batch_data_time if batch_data_time > 0 else 0

    entry = {
        "uttid": uttid,
        "llm_ms": round(llm_ms, 1),
        "fm_ms": round(fm_ms, 1),
        "voc_ms": round(voc_ms, 1),
        "total_ms": round(total_ms, 1),
        "gen_len": gen_len,
        "audio_sec": round(batch_data_time, 2),
        "rtf": round(rtf, 3),
        "hit_eos": hit_eos,
    }
    _timing_data.append(entry)
    print(f"[{len(_timing_data):2d}/50] {uttid:30s} | LLM:{llm_ms:7.0f}ms FM:{fm_ms:7.0f}ms VOC:{voc_ms:7.0f}ms | Total:{total_ms:8.0f}ms | Audio:{batch_data_time:5.1f}s RTF:{rtf:.3f}", flush=True)

    return [{}], {"batch_data_time": batch_data_time}

im.FunCineForgeInferModel.inference = _timed_inference


# ── Main ──
def main(**kwargs):
    if torch.cuda.is_available():
        _device = f"cuda:0"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    logger = get_logger(log_level=logging.INFO, local_rank=0, world_size=1)
    logger.info(f"Device: {_device}")

    output_dir = kwargs.get("output_dir")
    data_jsonl = kwargs.get("data_jsonl")

    lm_ckpt_path = kwargs.get("lm_ckpt_path", "")
    lm_exp_dir, lm_model_name, lm_ckpt_id, _ = lm_ckpt_path.rsplit("/", 3)
    logger.info(f"Loading LM from {lm_ckpt_path}")
    lm_model = AutoModel(
        model=os.path.join(lm_exp_dir, lm_model_name),
        init_param=lm_ckpt_path, output_dir=None, device=_device,
    )
    lm_model.model.to(dtype_map[kwargs.get("llm_dtype", "fp32")])

    fm_ckpt_path = kwargs.get("fm_ckpt_path", "")
    fm_exp_dir, fm_model_name, fm_ckpt_id, _ = fm_ckpt_path.rsplit("/", 3)
    logger.info(f"Loading FM from {fm_ckpt_path}")
    fm_model = AutoModel(
        model=os.path.join(fm_exp_dir, fm_model_name),
        init_param=fm_ckpt_path, output_dir=None, device=_device,
    )
    fm_model.model.to(dtype_map[kwargs.get("fm_dtype", "fp32")])

    voc_ckpt_path = kwargs.get("voc_ckpt_path", "")
    voc_exp_dir, voc_model_name, voc_ckpt_id, _ = voc_ckpt_path.rsplit("/", 3)
    logger.info(f"Loading VOC from {voc_ckpt_path}")
    voc_model = AutoModel(
        model=os.path.join(voc_exp_dir, voc_model_name),
        init_param=voc_ckpt_path, output_dir=None, device=_device,
    )
    voc_model.model.to(dtype_map[kwargs.get("voc_dtype", "fp32")])

    logger.info(f"Building inference model")
    kwargs["output_dir"] = output_dir
    kwargs["tokenizer"] = None
    model = AutoModel(
        **kwargs,
        lm_model=lm_model, fm_model=fm_model, voc_model=voc_model,
    )
    index_ds_class = tables.index_ds_classes.get(kwargs.get('index_ds'))
    dataset_conf = kwargs.get("dataset_conf")
    index_ds = index_ds_class(data_jsonl, **dataset_conf)

    print(f"\n{'='*100}", flush=True)
    print(f"FunCineForge MPS Benchmark — {len(index_ds)} items, device={_device}", flush=True)
    print(f"{'='*100}", flush=True)

    t_total_start = time.perf_counter()
    model.inference(input=index_ds, input_len=len(index_ds))
    t_total_end = time.perf_counter()

    # Summary
    print(f"\n{'='*100}", flush=True)
    print(f"BENCHMARK COMPLETE", flush=True)
    print(f"{'='*100}", flush=True)
    n = len(_timing_data)
    if n > 0:
        llm_avg = sum(d["llm_ms"] for d in _timing_data) / n
        fm_avg = sum(d["fm_ms"] for d in _timing_data) / n
        voc_avg = sum(d["voc_ms"] for d in _timing_data) / n
        total_avg = sum(d["total_ms"] for d in _timing_data) / n
        rtf_avg = sum(d["rtf"] for d in _timing_data) / n
        audio_total = sum(d["audio_sec"] for d in _timing_data)

        print(f"Items:       {n}")
        print(f"Wall time:   {(t_total_end - t_total_start):.1f}s")
        print(f"Audio total: {audio_total:.1f}s")
        print(f"")
        print(f"Per-item averages:")
        print(f"  LLM:   {llm_avg:8.1f} ms  ({llm_avg/total_avg*100:4.1f}%)")
        print(f"  FM:    {fm_avg:8.1f} ms  ({fm_avg/total_avg*100:4.1f}%)")
        print(f"  VOC:   {voc_avg:8.1f} ms  ({voc_avg/total_avg*100:4.1f}%)")
        print(f"  Total: {total_avg:8.1f} ms")
        print(f"  RTF:   {rtf_avg:.3f}")
        print(f"")
        print(f"LLM extremes: min={min(d['llm_ms'] for d in _timing_data):.0f}ms max={max(d['llm_ms'] for d in _timing_data):.0f}ms")
        print(f"FM extremes:  min={min(d['fm_ms'] for d in _timing_data):.0f}ms max={max(d['fm_ms'] for d in _timing_data):.0f}ms")
        print(f"RTF extremes: min={min(d['rtf'] for d in _timing_data):.3f} max={max(d['rtf'] for d in _timing_data):.3f}")

    # Save raw data
    with open("/tmp/mps_benchmark.json", "w") as f:
        json.dump(_timing_data, f, indent=2)
    print(f"\nRaw data saved to /tmp/mps_benchmark.json", flush=True)

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@hydra.main(config_path="decode_conf", config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    main(**kwargs)

if __name__ == "__main__":
    main_hydra()
