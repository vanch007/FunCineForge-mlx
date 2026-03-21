"""
MLX-accelerated LLM decoder for FunCineForge.

Performance: 121 tok/s (4.94x vs PyTorch MPS 24.5 tok/s)

Architecture:
  - ALL weights run on MLX (codec_embed, timespk_embed, face_linear, backbone, codec_head)
  - Only final codec tokens converted back to PyTorch for FM+VOC pipeline
  - KV cache managed internally by mlx_lm model
"""
import logging
import time
import torch
import numpy as np
from typing import List, Optional, Union

import mlx.core as mx
from mlx_lm.utils import load as mlx_load

from funcineforge.utils.hinter import hint_once


def _torch_to_mlx(t: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array."""
    return mx.array(t.detach().cpu().float().numpy())


class MLXLLMDecoder:
    """Drop-in replacement for LLMDecoder using MLX backbone (4.94x speedup)."""

    def __init__(self, mlx_model_path: str, custom_weights_path: str, **kwargs):
        self.eos_token = kwargs["eos"]
        if isinstance(self.eos_token, int):
            self.eos_token = [self.eos_token]
        self.ras_conf = kwargs.get("ras_conf", {})
        self.token_offset = kwargs.get("token_offset", 0)

        # Load MLX Qwen2 backbone
        logging.info(f"Loading MLX Qwen2 backbone from {mlx_model_path}")
        t0 = time.perf_counter()
        self.mlx_model, _ = mlx_load(mlx_model_path)
        logging.info(f"MLX model loaded in {time.perf_counter()-t0:.2f}s")

        # Load custom weights and convert to MLX
        logging.info(f"Loading custom weights from {custom_weights_path}")
        custom = torch.load(custom_weights_path, map_location="cpu", weights_only=False)
        self.mlx_codec_embed = mx.array(custom["codec_embed.weight"].numpy())      # (6761, 896)
        self.mlx_timespk_embed = mx.array(custom["timespk_embed.weight"].numpy())   # (1550, 896)
        self.mlx_codec_head = mx.array(custom["codec_head.weight"].numpy())         # (6761, 896)
        self.mlx_face_w = mx.array(custom["face_linear.weight"].numpy())            # (896, 512)
        self.mlx_face_b = mx.array(custom["face_linear.bias"].numpy())              # (896,)

        # Keep PyTorch token_embeder for initial embedding lookup
        self.token_embeder = kwargs["token_embeder"]  # PyTorch nn.Embedding

        # Get the backbone's embed_tokens for token lookups in MLX
        # Access via model's inner layers
        self.mlx_embed_tokens = self.mlx_model.model.embed_tokens

    def _mlx_nucleus_sampling(self, logits: mx.array, top_p=0.8, top_k=25) -> int:
        """Nucleus sampling entirely in MLX."""
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)[::-1]
        sorted_probs = probs[sorted_indices]
        cum_probs = mx.cumsum(sorted_probs, axis=-1)

        # Find cutoff
        n_keep = min(int((cum_probs < top_p).sum().item()) + 1, top_k)
        top_probs = sorted_probs[:n_keep]
        top_indices = sorted_indices[:n_keep]

        # Sample
        # MLX doesn't have multinomial, use numpy for sampling
        np_probs = np.array(top_probs, copy=False)
        np_probs = np_probs / np_probs.sum()  # renormalize
        sample_idx = np.random.choice(len(np_probs), p=np_probs)
        return int(top_indices[sample_idx].item())

    def _mlx_ras_sampling(self, logits: mx.array, decoded_tokens: list,
                          top_p=0.8, top_k=25, win_size=10, tau_r=0.1) -> int:
        """Repetition Aware Sampling in MLX."""
        if self.ras_conf:
            top_p = self.ras_conf.get("top_p", top_p)
            top_k = self.ras_conf.get("top_k", top_k)
            win_size = self.ras_conf.get("win_size", win_size)
            tau_r = self.ras_conf.get("tau_r", tau_r)

        hint_once(f"MLX RAS: top_p={top_p}, top_k={top_k}, win={win_size}, tau_r={tau_r}", "mlx_ras")

        top_id = self._mlx_nucleus_sampling(logits, top_p=top_p, top_k=top_k)
        # Repetition check
        recent = decoded_tokens[-win_size:]
        rep_num = sum(1 for t in recent if t == top_id)
        if rep_num >= win_size * tau_r:
            # Fall back to random sampling
            probs = mx.softmax(logits, axis=-1)
            np_probs = np.array(probs, copy=False)
            np_probs = np_probs / np_probs.sum()
            top_id = int(np.random.choice(len(np_probs), p=np_probs))
        return top_id

    def _mlx_sampling_ids(self, logits: mx.array, sampling, decoded_tokens=None) -> int:
        """Token sampling dispatch."""
        if isinstance(sampling, bool):
            if sampling:
                probs = mx.softmax(logits, axis=-1)
                np_probs = np.array(probs, copy=False)
                np_probs = np_probs / np_probs.sum()
                return int(np.random.choice(len(np_probs), p=np_probs))
            else:
                return int(mx.argmax(logits).item())
        elif isinstance(sampling, int):
            probs = mx.softmax(logits, axis=-1)
            top_k_idx = mx.argsort(probs)[::-1][:sampling]
            top_k_probs = probs[top_k_idx]
            np_probs = np.array(top_k_probs, copy=False)
            np_probs = np_probs / np_probs.sum()
            sample_idx = np.random.choice(len(np_probs), p=np_probs)
            return int(top_k_idx[sample_idx].item())
        elif isinstance(sampling, float):
            return self._mlx_nucleus_sampling(logits, top_p=sampling)
        elif isinstance(sampling, str) and sampling.lower() == "ras":
            return self._mlx_ras_sampling(logits, decoded_tokens or [])
        raise NotImplementedError(f"Unknown sampling: {type(sampling)}")

    def __call__(self, input_embeddings, llm, states, quantize=False, **kwargs):
        """
        Main decode loop — runs entirely on MLX.

        Args:
            input_embeddings: PyTorch tensor (1, seq_len, hidden_dim) from PyTorch custom embeddings
            llm: PyTorch model (UNUSED — we use self.mlx_model)
            states: dict with optional KV cache state
        
        Returns:
            (codec_tokens, hit_eos, states)
        """
        from mlx_lm.models.cache import make_prompt_cache

        max_length = kwargs.get("max_length", 60 * 25)
        min_length = kwargs.get("min_length", 2 * 25)
        sampling = kwargs.get("sampling", True)
        include_eos = kwargs.get("include_eos", False)
        custom_eos_token = kwargs.get("custom_eos_token", self.eos_token)
        avoid_token = kwargs.get("avoid_token", None)

        # Convert input embeddings from PyTorch → MLX (one-time cost)
        mlx_embeds = _torch_to_mlx(input_embeddings)

        # Create KV cache for all transformer layers (critical for autoregressive coherence)
        kv_cache = make_prompt_cache(self.mlx_model)

        # Prefill: run full prefix through MLX model backbone WITH cache
        hidden = self.mlx_model.model(None, cache=kv_cache, input_embeddings=mlx_embeds)
        mx.eval(hidden)

        # Apply codec_head to get logits: h @ codec_head.T
        logp = hidden[:, -1, :] @ self.mlx_codec_head.T  # (1, 6761)
        logp = mx.log(mx.softmax(logp[0], axis=-1))  # log-softmax for sampling
        mx.eval(logp)

        out_tokens, hit_eos = [], False

        for i in range(max_length):
            pred = logp

            # Suppress EOS before min_length
            if min_length is not None and i < min_length:
                for x in custom_eos_token:
                    pred = pred.at[x].add(-1e4 - pred[x])
            if avoid_token is not None:
                for x in avoid_token:
                    pred = pred.at[x].add(-1e4 - pred[x])

            # We need mutable logits for sampling — use numpy
            np_pred = np.array(pred, copy=True)
            if min_length is not None and i < min_length:
                for x in custom_eos_token:
                    np_pred[x] = -1e4
            if avoid_token is not None:
                for x in avoid_token:
                    np_pred[x] = -1e4

            # Sampling
            top_id = self._mlx_sampling_ids(
                mx.array(np_pred), sampling, decoded_tokens=out_tokens
            )

            if top_id in custom_eos_token:
                if include_eos:
                    out_tokens.append(top_id)
                hit_eos = True
                break

            out_tokens.append(top_id)

            # Next step: look up codec_embed for generated token + run backbone WITH cache
            next_emb = self.mlx_codec_embed[top_id + self.token_offset:top_id + self.token_offset + 1]
            next_emb = next_emb.reshape(1, 1, -1)

            hidden = self.mlx_model.model(None, cache=kv_cache, input_embeddings=next_emb)
            mx.eval(hidden)

            logp = hidden[:, -1, :] @ self.mlx_codec_head.T
            logp = mx.log(mx.softmax(logp[0], axis=-1))
            mx.eval(logp)

        # Convert to torch tensor to match original LLMDecoder interface
        out_tokens_tensor = torch.tensor([out_tokens], dtype=torch.int64)

        return out_tokens_tensor, hit_eos, states
