# SPDX-License-Identifier: Apache-2.0
"""Hybrid handoffs: Qwen3.8 Flash Next's whole-prefix state in place of locally prefilled caches.

The model mixes Gated DeltaNet layers, whose recurrent state is only valid for a whole prefix,
with QSA sparse-attention layers, whose keys, values and raw index keys grow per token, and one
PLE layer with a short-conv state. So a hybrid handoff always covers tokens [0, end): the
producer exports every layer's state at `end`, and fresh caches replace whatever the local
prefix cache restored. Each tensor arrives as its own manifest entry in a fixed layout:

    qsa  k, v        [T, kv_heads, head_dim]           post-RoPE, activation dtype
    qsa  index_keys  [T, indexer_head_dim]             raw: pre-norm, pre-RoPE
    gdn  conv        [conv_kernel - 1, conv_dim]       oldest tap first; channels q | k | v
    gdn  ssm         [v_heads, v_dim, k_dim]           float32
    ple  conv        [(kernel - 1) * dilation, hc_dim] oldest tap first
    mtp  k, v        [T - 1, kv_heads, head_dim]       the MTP head's QSA layer (index = layers)
    mtp  index_keys  [T - 1, indexer_head_dim]
    mtp  hidden      [1, hc_dim]                       trunk hidden streams of token T - 1

The mtp entries come only from a producer that runs MTP speculative decoding. Row t of the
head's layer pairs the trunk's hidden state at t with token t + 1, exactly what Lightning MTP's
prompt priming folds, so with the hidden state of token T - 1 pending they resume priming at
`end` like a prefix-cache sidecar does. Without them the head starts on the local suffix only.

Text-only prompts index positions 0..T-1, and the PLE layer's n-gram context is the prompt's
last tokens, so neither crosses the link.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from . import wire
from .job import Chunk, PrefillResult
from .receiver import HandoffError

MODEL_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})


@dataclass(frozen=True)
class HybridGeometry:
    """Which layer holds which state, and every exported tensor's shape and dtype."""

    layers: int
    kinds: tuple[str, ...]  # per layer: "qsa" or "gdn"
    ple_layer: int  # the layer that also carries PLE state, or -1
    shapes: dict[tuple[str, str], tuple[int, ...]]  # (kind, name) -> exported shape
    ple_context: int
    mtp_layer: int = -1  # the MTP head's layer index when the local model drafts with it

    def expected(self) -> dict[tuple[int, str, str], tuple[int, ...]]:
        """Every entry a complete handoff carries, keyed by (layer, kind, name)."""
        out = {}
        for index, kind in enumerate(self.kinds):
            names = ("k", "v", "index_keys") if kind == "qsa" else ("conv", "ssm")
            for name in names:
                out[(index, kind, name)] = self.shapes[(kind, name)]
        if self.ple_layer >= 0:
            out[(self.ple_layer, "ple", "conv")] = self.shapes[("ple", "conv")]
        return out

    def expected_mtp(self) -> dict[tuple[int, str, str], tuple[int, ...]]:
        """The optional MTP-head entries; -1 rows means one fewer than the prompt."""
        if self.mtp_layer < 0:
            return {}
        return {(self.mtp_layer, "mtp", name): self.shapes[("mtp", name)]
                for name in ("k", "v", "index_keys", "hidden")}


def _text_config(model: Any) -> Any:
    for holder in (model, getattr(model, "language_model", None)):
        for name in ("args", "config"):
            config = getattr(holder, name, None)
            if config is not None:
                return getattr(config, "text_config", None) or config
    return None


def _get(config: Any, name: str) -> Any:
    return config.get(name) if isinstance(config, dict) else getattr(config, name, None)


def _mtp_host(model: Any) -> Any:
    """The model object Lightning MTP drafts with, or None when MTP is off."""
    try:
        from omlx.patches.mlx_lm_mtp import prompt_priming
    except Exception:
        return None
    if not prompt_priming.priming_enabled():
        return None
    host = prompt_priming._eligible_host(model)
    return host if host is not None and callable(getattr(host, "make_mtp_cache", None)) else None


def hybrid_geometry(model: Any, caches: list[Any]) -> tuple[HybridGeometry | None, str]:
    """The model's hybrid layout, or None and why the model cannot take a hybrid handoff."""
    config = _text_config(model)
    if config is None or _get(config, "model_type") not in MODEL_TYPES:
        return None, "the model is not a supported hybrid"
    layer_types = list(_get(config, "layer_types") or [])
    if len(layer_types) != len(caches):
        return None, f"the config lists {len(layer_types)} layer types for {len(caches)} caches"
    kinds = []
    for index, (layer_type, cache) in enumerate(zip(layer_types, caches)):
        name = type(cache).__name__
        if layer_type == "linear_attention" and name == "ArraysCache":
            kinds.append("gdn")
        elif layer_type in ("full_attention", "qwen_sparse_attention") and name == "QSAKVCache":
            kinds.append("qsa")
        else:
            return None, f"layer {index} is {layer_type} with a {name}"
    ple_layers = [index for index, cache in enumerate(caches) if kinds[index] == "gdn" and len(cache.cache) == 4]
    if len(ple_layers) > 1:
        return None, "more than one PLE layer"
    k_heads, v_heads = _get(config, "linear_num_key_heads"), _get(config, "linear_num_value_heads")
    k_dim, v_dim = _get(config, "linear_key_head_dim"), _get(config, "linear_value_head_dim")
    kernel = _get(config, "linear_conv_kernel_dim")
    kv_heads, head_dim = _get(config, "num_key_value_heads"), _get(config, "head_dim")
    index_dim = _get(config, "indexer_head_dim")
    ple_kernel, dilation = _get(config, "ple_conv_kernel_size"), _get(config, "ngram_size")
    hc_dim = (_get(config, "hc_count") or 1) * (_get(config, "hidden_size") or 0)
    sizes = (k_heads, v_heads, k_dim, v_dim, kernel, kv_heads, head_dim, index_dim)
    if not all(isinstance(size, int) and size > 0 for size in sizes):
        return None, "the config lacks the hybrid layer sizes"
    shapes = {
        ("qsa", "k"): (-1, kv_heads, head_dim),
        ("qsa", "v"): (-1, kv_heads, head_dim),
        ("qsa", "index_keys"): (-1, index_dim),
        ("gdn", "conv"): (kernel - 1, 2 * k_heads * k_dim + v_heads * v_dim),
        ("gdn", "ssm"): (v_heads, v_dim, k_dim),
    }
    ple_layer = ple_layers[0] if ple_layers else -1
    if ple_layer >= 0:
        if not (ple_kernel and dilation and hc_dim):
            return None, "the config lacks the PLE sizes"
        shapes[("ple", "conv")] = ((ple_kernel - 1) * dilation, hc_dim)
    mtp_layer = len(caches) if hc_dim and _mtp_host(model) is not None else -1
    if mtp_layer >= 0:
        shapes[("mtp", "k")] = shapes[("qsa", "k")]
        shapes[("mtp", "v")] = shapes[("qsa", "v")]
        shapes[("mtp", "index_keys")] = shapes[("qsa", "index_keys")]
        shapes[("mtp", "hidden")] = (1, hc_dim)
    return (
        HybridGeometry(len(caches), tuple(kinds), ple_layer, shapes, max(0, (dilation or 1) - 1), mtp_layer),
        "",
    )


def check_manifest(manifest: wire.Manifest, geometry: HybridGeometry) -> None:
    """Refuse a hybrid manifest unless it carries exactly the model's entries, whole prefix."""
    if not manifest.hybrid:
        raise HandoffError("the producer sent attention pages for a hybrid model")
    if manifest.tp_size != 1:
        raise HandoffError("hybrid handoffs are single-rank")
    tokens = manifest.prompt_tokens
    have = {(entry.layer, entry.kind, entry.name): entry for entry in manifest.layers if entry.kind != "mtp"}
    mtp = {(entry.layer, entry.kind, entry.name): entry for entry in manifest.layers if entry.kind == "mtp"}
    want = geometry.expected()
    want_mtp = geometry.expected_mtp()
    # The MTP head's history is optional on both sides; when the model drafts with it, it must be whole.
    if mtp and want_mtp:
        if set(mtp) != set(want_mtp):
            raise HandoffError(f"MTP entries differ: got {sorted(mtp)[:4]}, expected {sorted(want_mtp)}")
        for key, shape in want_mtp.items():
            full = tuple(tokens - 1 if size == -1 else size for size in shape)
            if mtp[key].shape != full:
                raise HandoffError(f"entry {key} is {list(mtp[key].shape)}, expected {list(full)}")
    if set(have) != set(want):
        missing = sorted(set(want) - set(have))[:3]
        extra = sorted(set(have) - set(want))[:3]
        raise HandoffError(f"hybrid entries differ: missing {missing}, unexpected {extra}")
    for key, shape in want.items():
        entry = have[key]
        full = tuple(tokens if size == -1 else size for size in shape)
        if entry.shape != full:
            raise HandoffError(f"entry {key} is {list(entry.shape)}, expected {list(full)}")
        if key[2] == "ssm" and entry.dtype != "float32":
            raise HandoffError(f"entry {key} must be float32")


def _tensors(mx: Any, result: PrefillResult, mtp: bool = False) -> dict[tuple[int, str, str], Any]:
    """Each entry's chunks joined in row order and viewed as its exported shape and dtype.

    Backbone entries by default, the MTP head's with `mtp`."""
    grouped: dict[int, list[Chunk]] = defaultdict(list)
    entries = {}
    for chunk in result.chunks:
        if (chunk.layer.kind == "mtp") != mtp:
            continue
        grouped[chunk.layer.index].append(chunk)
        entries[chunk.layer.index] = chunk.layer
    out = {}
    for index, chunks in grouped.items():
        entry = entries[index]
        chunks.sort(key=lambda chunk: chunk.row_start)
        rows = sum(chunk.rows for chunk in chunks)
        if rows != entry.shape[0] or any(
            a.row_start + a.rows != b.row_start for a, b in zip(chunks, chunks[1:])
        ):
            raise HandoffError(f"entry {entry.layer}/{entry.kind}/{entry.name} arrived incomplete")
        flat = chunks[0].data if len(chunks) == 1 else mx.concatenate([c.data for c in chunks])
        out[(entry.layer, entry.kind, entry.name)] = flat.view(
            getattr(mx, entry.dtype)
        ).reshape(entry.shape)
    return out


def build_caches(
    mx: Any,
    make_cache: Any,
    geometry: HybridGeometry,
    result: PrefillResult,
    tokens: list[int],
    dtype: Any,
) -> list[Any]:
    """Fresh caches holding the handoff's state for prompt tokens [0, len(tokens))."""
    end = len(tokens)
    if result.manifests[0].prompt_tokens != end:
        raise HandoffError("the handoff covers a different prefix")
    tensors = _tensors(mx, result)
    caches = make_cache()
    positions = mx.arange(end, dtype=mx.int32)[None]
    for index, kind in enumerate(geometry.kinds):
        cache = caches[index]
        if kind == "qsa":
            keys = tensors[(index, "qsa", "k")].transpose(1, 0, 2)[None].astype(dtype)
            values = tensors[(index, "qsa", "v")].transpose(1, 0, 2)[None].astype(dtype)
            index_keys = tensors[(index, "qsa", "index_keys")][None].astype(dtype)
            cache.state = (keys, values, index_keys, positions)
        else:
            cache[0] = tensors[(index, "gdn", "conv")][None].astype(dtype)
            cache[1] = tensors[(index, "gdn", "ssm")][None].astype(mx.float32)
    if geometry.ple_layer >= 0:
        cache = caches[geometry.ple_layer]
        cache[2] = tensors[(geometry.ple_layer, "ple", "conv")][None].astype(dtype)
        context = tokens[-geometry.ple_context :] if geometry.ple_context else []
        cache[3] = mx.array([context], dtype=mx.int64)
    mx.eval([array for cache in caches for array in _arrays(cache)])
    return caches


def build_mtp(
    mx: Any,
    model: Any,
    geometry: HybridGeometry,
    result: PrefillResult,
    end: int,
    dtype: Any,
) -> tuple[list[Any], Any] | None:
    """The MTP head's cache over pairs [0, end - 1) and the pending trunk hidden of token end - 1.

    None when the handoff carries no MTP history or the local model does not draft with it."""
    if geometry.mtp_layer < 0 or end < 2:
        return None
    tensors = _tensors(mx, result, mtp=True)
    layer = geometry.mtp_layer
    if not tensors or set(tensors) != set(geometry.expected_mtp()):
        return None
    host = _mtp_host(model)
    if host is None:
        return None
    caches = host.make_mtp_cache()
    if len(caches) != 1:
        raise HandoffError(f"the MTP head has {len(caches)} layers; the handoff carries one")
    rows = end - 1
    keys = tensors[(layer, "mtp", "k")].transpose(1, 0, 2)[None].astype(dtype)
    values = tensors[(layer, "mtp", "v")].transpose(1, 0, 2)[None].astype(dtype)
    index_keys = tensors[(layer, "mtp", "index_keys")][None].astype(dtype)
    caches[0].state = (keys, values, index_keys, mx.arange(rows, dtype=mx.int32)[None])
    hidden = tensors[(layer, "mtp", "hidden")][None].astype(dtype)  # [1, 1, hc_dim]
    mx.eval([*_arrays(caches[0]), hidden])
    return caches, hidden


def _arrays(cache: Any) -> list[Any]:
    state = cache.state
    if isinstance(state, tuple) and len(state) == 3 and isinstance(state[0], list):
        state = state[0]  # ArraysCache: (arrays, left_padding, lengths)
    return [item for item in state if item is not None and hasattr(item, "shape")]
