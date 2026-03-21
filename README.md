### <p align="center">「English | [简体中文](./README_zh.md)」</p>

<p align="center">
<b>🎬 FunCineForge-MLX: Apple Silicon Accelerated Movie Dubbing</b><br>
<i>MLX-optimized fork of <a href="https://github.com/FunAudioLLM/FunCineForge">FunCineForge</a> — 4.76x LLM speedup on Apple Silicon</i>
</p>

<div align="center">

![license](https://img.shields.io/github/license/modelscope/modelscope.svg)
<a href=""><img src="https://img.shields.io/badge/OS-macOS-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.10-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/MLX->=0.31-green"></a>
<a href=""><img src="https://img.shields.io/badge/Pytorch->=2.1-blue"></a>

</div>

<div align="center">  
<h4><a href="#performance">Performance</a>
｜<a href="#quickstart">Quick Start</a>
｜<a href="#changes">What Changed</a>
｜<a href="#credits">Credits</a>
</h4>
</div>

This fork optimizes [FunCineForge](https://github.com/FunAudioLLM/FunCineForge) for **Apple Silicon (M1/M2/M3/M4)** by replacing the PyTorch LLM backbone with [MLX](https://github.com/ml-explore/mlx), achieving significant inference speedups while maintaining full stability (50/50 runs, zero crashes).

<a name="performance"></a>
## Performance 🚀

Benchmarked on MacBook Pro M3 Max (128GB), Qwen2-0.5B backbone, fp32:

| Stage | PyTorch MPS | MLX | Speedup |
|-------|:-----------:|:---:|:-------:|
| **LLM decode** | 24.5 tok/s | **116.6 tok/s** | **4.76x** |
| LLM per item | 15,312 ms | **497 ms** | **30.8x** |
| Stability | 50/50 ✅ | **50/50 ✅** | — |

### MPS Stability Fixes (included)

| Fix | Issue Solved |
|-----|-------------|
| SDPA → eager attention | Metal memory corruption |
| autocast → nullcontext | Metal assertion failures |
| torchaudio → soundfile | TorchCodec crash on MPS |
| MPS cache flush per item | Cumulative memory pressure |

<a name="quickstart"></a>
## Quick Start ⚡

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Conda + Python 3.10

### Installation

```bash
git clone https://github.com/vanch007/FunCineForge-mlx.git
cd FunCineForge-mlx
conda create -n FunCineForge python=3.10 -y && conda activate FunCineForge

# Install base dependencies
python setup.py

# Install MLX (Apple Silicon only)
pip install mlx mlx-lm
```

### Inference

```bash
cd exps
bash infer.sh
```

The MLX model (`vanch007/FunCineForge-mlx-qwen2`) is **auto-downloaded** from HuggingFace on first run.

### Configuration

In `exps/decode_conf/decode.yaml`:

```yaml
# Enable MLX LLM acceleration (default: true)
use_mlx: true

# Other settings
llm_dtype: fp32    # fp32 only on MPS (fp16 causes Metal crashes)
fm_dtype: fp32
voc_dtype: fp32
```

Set `use_mlx: false` to fall back to PyTorch MPS.

<a name="changes"></a>
## What Changed vs Upstream 🔧

### New Files

| File | Description |
|------|-------------|
| `funcineforge/models/utils/mlx_llm_decoder.py` | Full MLX decode loop — backbone + codec_head + sampling |

### Modified Files

| File | Change |
|------|--------|
| `funcineforge/models/language_model.py` | `use_mlx` toggle with auto-download from HuggingFace; MPS stability fixes (eager attention, transformers 5.x compat) |
| `funcineforge/models/utils/llm_decoding.py` | Safe float constants for MPS fp16 edge cases |
| `exps/decode_conf/decode.yaml` | `use_mlx: true` default |

### Architecture

```
PyTorch (data loader) → MLX (Qwen2 backbone + codec_head) → PyTorch (Flow Matching + Vocoder)
                              ↑
                    Auto-downloaded from HuggingFace:
                    vanch007/FunCineForge-mlx-qwen2
```

The MLX decoder runs the **entire LLM pipeline in MLX** — including custom embeddings (codec_embed, timespk_embed), the Qwen2 transformer backbone, and the codec_head projection. Only the final codec token sequence is converted back to PyTorch for the Flow Matching and Vocoder stages.

### MLX Model

Hosted at: [vanch007/FunCineForge-mlx-qwen2](https://huggingface.co/vanch007/FunCineForge-mlx-qwen2)

| File | Size | Description |
|------|:----:|-------------|
| `model.safetensors` | 1.98 GB | MLX Qwen2-0.5B backbone (fp32) |
| `custom_weights.pt` | 55.9 MB | codec_embed, timespk_embed, codec_head, face_linear |

<a name="credits"></a>
## Credits 🙏

- **[FunCineForge](https://github.com/FunAudioLLM/FunCineForge)** — Original project by Tongyi Lab Speech Team & USTC
- **[MLX](https://github.com/ml-explore/mlx)** — Apple's machine learning framework for Apple Silicon
- **[mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms)** — MLX language model utilities

## Publication 📚

If you use this code, please cite the original FunCineForge paper:

```bibtex
@misc{liu2026funcineforgeunifieddatasettoolkit,
    title={FunCineForge: A Unified Dataset Toolkit and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes}, 
    author={Jiaxuan Liu and Yang Xiang and Han Zhao and Xiangang Li and Zhenhua Ling},
    year={2026},
    eprint={2601.14777},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
}
```

## Disclaimer

⚠️ This is a community fork for research purposes. Not affiliated with Tongyi Lab.

⚠️ MLX acceleration is Apple Silicon only. For CUDA GPUs, use the [upstream repo](https://github.com/FunAudioLLM/FunCineForge).