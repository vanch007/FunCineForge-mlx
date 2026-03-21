### <p align="center">「[English](./README.md) | 简体中文」</p>

<p align="center">
<b>🎬 FunCineForge-MLX：Apple Silicon 加速的影视配音模型</b><br>
<i>基于 <a href="https://github.com/FunAudioLLM/FunCineForge">FunCineForge</a> 的 MLX 优化分支 — Apple Silicon 上 LLM 推理加速 4.76 倍</i>
</p>

<div align="center">

![license](https://img.shields.io/github/license/modelscope/modelscope.svg)
<a href=""><img src="https://img.shields.io/badge/OS-macOS-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.10-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/MLX->=0.31-green"></a>
<a href=""><img src="https://img.shields.io/badge/Pytorch->=2.1-blue"></a>

</div>

<div align="center">
<h4><a href="#性能">性能</a>
｜<a href="#快速开始">快速开始</a>
｜<a href="#改动说明">改动说明</a>
｜<a href="#致谢">致谢</a>
</h4>
</div>

本分支将 [FunCineForge](https://github.com/FunAudioLLM/FunCineForge) 针对 **Apple Silicon (M1/M2/M3/M4)** 进行优化，使用 [MLX](https://github.com/ml-explore/mlx) 替换 PyTorch LLM 骨干网络推理，实现显著加速，同时保持完全稳定（50/50 次运行零崩溃）。

<a name="性能"></a>
## 性能 🚀

测试环境：MacBook Pro M3 Max (128GB)，Qwen2-0.5B 骨干网络，fp32：

| 阶段 | PyTorch MPS | MLX | 加速比 |
|------|:-----------:|:---:|:-----:|
| **LLM 解码** | 24.5 tok/s | **116.6 tok/s** | **4.76x** |
| LLM 单项耗时 | 15,312 ms | **497 ms** | **30.8x** |
| 稳定性 | 50/50 ✅ | **50/50 ✅** | — |

### MPS 稳定性修复（已包含）

| 修复 | 解决的问题 |
|------|-----------|
| SDPA → eager attention | Metal 内存腐蚀 |
| autocast → nullcontext | Metal 断言失败 |
| torchaudio → soundfile | TorchCodec MPS 崩溃 |
| 每项推理后刷新 MPS 缓存 | 累积内存压力 |

<a name="快速开始"></a>
## 快速开始 ⚡

### 前置条件

- macOS + Apple Silicon (M1/M2/M3/M4)
- Conda + Python 3.10

### 安装

```bash
git clone https://github.com/vanch007/FunCineForge-mlx.git
cd FunCineForge-mlx
conda create -n FunCineForge python=3.10 -y && conda activate FunCineForge

# 安装基础依赖
python setup.py

# 安装 MLX（仅限 Apple Silicon）
pip install mlx mlx-lm
```

### 推理

```bash
cd exps
bash infer.sh
```

首次运行时会自动从 HuggingFace 下载 MLX 模型 (`vanch007/FunCineForge-mlx-qwen2`)。

### 配置

在 `exps/decode_conf/decode.yaml` 中：

```yaml
# 启用 MLX LLM 加速（默认已开启）
use_mlx: true

# 其他设置
llm_dtype: fp32    # MPS 上仅支持 fp32（fp16 会导致 Metal 崩溃）
fm_dtype: fp32
voc_dtype: fp32
```

将 `use_mlx` 设为 `false` 可回退到 PyTorch MPS。

<a name="改动说明"></a>
## 相对上游的改动 🔧

### 新增文件

| 文件 | 说明 |
|------|------|
| `funcineforge/models/utils/mlx_llm_decoder.py` | 完整的 MLX 解码循环 — 骨干网络 + codec_head + 采样 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `funcineforge/models/language_model.py` | `use_mlx` 开关 + HuggingFace 自动下载；MPS 稳定性修复（eager attention, transformers 5.x 兼容） |
| `funcineforge/models/utils/llm_decoding.py` | MPS fp16 边界情况安全常量修复 |
| `exps/decode_conf/decode.yaml` | 默认 `use_mlx: true` |

### 架构

```
PyTorch（数据加载） → MLX（Qwen2 骨干 + codec_head） → PyTorch（Flow Matching + 声码器）
                           ↑
                 自动从 HuggingFace 下载：
                 vanch007/FunCineForge-mlx-qwen2
```

MLX 解码器在 MLX 中运行**整个 LLM 管线** — 包括自定义嵌入层（codec_embed、timespk_embed）、Qwen2 Transformer 骨干和 codec_head 投影层。仅将最终的 codec token 序列转回 PyTorch 用于 Flow Matching 和声码器阶段。

### MLX 模型

托管于：[vanch007/FunCineForge-mlx-qwen2](https://huggingface.co/vanch007/FunCineForge-mlx-qwen2)

| 文件 | 大小 | 说明 |
|------|:----:|------|
| `model.safetensors` | 1.98 GB | MLX Qwen2-0.5B 骨干权重 (fp32) |
| `custom_weights.pt` | 55.9 MB | codec_embed, timespk_embed, codec_head, face_linear |

<a name="致谢"></a>
## 致谢 🙏

- **[FunCineForge](https://github.com/FunAudioLLM/FunCineForge)** — 原始项目，通义实验室语音团队 & 中国科大
- **[MLX](https://github.com/ml-explore/mlx)** — Apple 的 Apple Silicon 机器学习框架
- **[mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms)** — MLX 语言模型工具库

## 引用 📚

如果您使用了本代码，请引用原始 FunCineForge 论文：

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

## 免责声明

⚠️ 这是社区维护的研究分支，与通义实验室无关。

⚠️ MLX 加速仅支持 Apple Silicon。如需使用 CUDA GPU，请前往[上游仓库](https://github.com/FunAudioLLM/FunCineForge)。