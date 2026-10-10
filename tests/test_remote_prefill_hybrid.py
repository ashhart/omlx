# SPDX-License-Identifier: Apache-2.0
"""Hybrid handoffs: manifests of named whole-prefix entries are checked and become fresh caches."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.remote_prefill import hybrid, wire
from omlx.remote_prefill.job import Chunk, PrefillResult
from omlx.remote_prefill.receiver import HandoffError

TOKENS = list(range(100, 107))  # 7 prompt tokens


class QSAKVCache:
    """Stands in for the vendored QSA cache: state is (keys, values, index_keys, positions)."""

    def __init__(self):
        self.state = (None, None, None, None)


class ArraysCache:
    def __init__(self, size):
        self.cache = [None] * size

    def __setitem__(self, index, value):
        self.cache[index] = value

    def __getitem__(self, index):
        return self.cache[index]

    @property
    def state(self):
        return self.cache, None, None


CONFIG = SimpleNamespace(
    model_type="qwen4_exp_text",
    layer_types=["linear_attention", "linear_attention", "full_attention"],
    linear_num_key_heads=2, linear_num_value_heads=3, linear_key_head_dim=4, linear_value_head_dim=5,
    linear_conv_kernel_dim=4, num_key_value_heads=2, head_dim=8, indexer_head_dim=6,
    ple_conv_kernel_size=4, ngram_size=3, hc_count=2, hidden_size=5,
)
MODEL = SimpleNamespace(args=CONFIG)


def make_cache():
    return [ArraysCache(2), ArraysCache(4), QSAKVCache()]


def geometry():
    found, reason = hybrid.hybrid_geometry(MODEL, make_cache())
    assert found is not None, reason
    return found


def entries_and_chunks(split=None):
    """Every entry with distinct values, the qsa keys optionally split into two frames."""
    spec = {
        (0, "gdn", "conv"): ((3, 2 * 2 * 4 + 3 * 5), "bfloat16", ["block", "channel"]),
        (0, "gdn", "ssm"): ((3, 5, 4), "float32", ["block", "v_dim", "k_dim"]),
        (1, "gdn", "conv"): ((3, 31), "bfloat16", ["block", "channel"]),
        (1, "gdn", "ssm"): ((3, 5, 4), "float32", ["block", "v_dim", "k_dim"]),
        (1, "ple", "conv"): ((9, 10), "bfloat16", ["block", "channel"]),
        (2, "qsa", "k"): ((7, 2, 8), "bfloat16", ["block", "head", "head_dim"]),
        (2, "qsa", "v"): ((7, 2, 8), "bfloat16", ["block", "head", "head_dim"]),
        (2, "qsa", "index_keys"): ((7, 6), "bfloat16", ["block", "index_dim"]),
    }
    values, entries, chunks = {}, [], []
    for position, (key, (shape, dtype, dims)) in enumerate(spec.items()):
        layer, kind, name = key
        array = (mx.arange(int(mx.prod(mx.array(shape)).item())) + 1000 * position).reshape(shape)
        array = array.astype(getattr(mx, dtype))
        values[key] = array
        entry = wire.LayerExport.from_dict({"index": position, "kind": kind, "layer": layer, "name": name,
                                            "shape": list(shape), "dims": dims, "dtype": dtype})
        entries.append(entry)
        carrier = mx.uint32 if dtype == "float32" else mx.uint16
        flat = array.view(carrier)
        cuts = [0, split, shape[0]] if split and key == (2, "qsa", "k") else [0, shape[0]]
        for start, stop in zip(cuts, cuts[1:]):
            chunks.append(Chunk(0, entry, start, stop - start, flat[start:stop].reshape(-1)))
    return values, entries, chunks


def manifest(entries, **changes):
    fields = dict(handoff="h", model="m", prompt_tokens=len(TOKENS), first_token=0,
                  token_sha256=wire.token_sha256(TOKENS), block_size=1, tp_rank=0, tp_size=1,
                  layers=tuple(entries), frames=len(entries))
    fields.update(changes)
    return wire.Manifest(**fields)


def test_a_hybrid_manifest_parses_and_must_cover_the_whole_prefix():
    _, entries, _ = entries_and_chunks()
    raw = {"protocol": 1, "handoff": "h", "model": "m", "prompt_tokens": 7, "first_token": 0,
           "token_sha256": "x", "block_size": 1, "tp_rank": 0, "tp_size": 1, "frames": 8,
           "layers": [{"index": e.index, "kind": e.kind, "layer": e.layer, "name": e.name,
                       "shape": list(e.shape), "dims": list(e.dims), "dtype": e.dtype} for e in entries]}
    parsed = wire.Manifest.from_json(json.dumps(raw).encode())
    assert parsed.hybrid and parsed.layers[0].name == "conv"
    with pytest.raises(wire.WireError, match="whole prefix"):
        wire.Manifest.from_json(json.dumps({**raw, "first_token": 3}).encode())
    attention = {"index": 9, "kind": "attention", "shape": [1, 16, 2, 8], "dims": ["block", "token", "head", "head_dim"],
                 "dtype": "bfloat16", "heads": 2, "total_heads": 2, "head_size": 8}
    with pytest.raises(wire.WireError, match="mixes"):
        wire.Manifest.from_json(json.dumps({**raw, "layers": raw["layers"] + [attention]}).encode())
    with pytest.raises(wire.WireError, match="model layer and name"):
        wire.LayerExport.from_dict({**raw["layers"][0], "name": ""})
    with pytest.raises(wire.WireError, match="dims"):
        wire.LayerExport.from_dict({**raw["layers"][0], "dims": ["channel", "block"]})


def test_the_geometry_follows_the_layer_types_and_finds_the_ple_layer():
    found = geometry()
    assert found.kinds == ("gdn", "gdn", "qsa") and found.ple_layer == 1 and found.ple_context == 2
    assert found.shapes[("gdn", "conv")] == (3, 31) and found.shapes[("ple", "conv")] == (9, 10)
    other = SimpleNamespace(args=SimpleNamespace(**{**vars(CONFIG), "model_type": "llama"}))
    assert hybrid.hybrid_geometry(other, make_cache())[0] is None
    wrong = [ArraysCache(2), QSAKVCache(), QSAKVCache()]
    assert "layer 1" in hybrid.hybrid_geometry(MODEL, wrong)[1]


def test_check_manifest_wants_exactly_the_models_entries():
    found = geometry()
    _, entries, _ = entries_and_chunks()
    hybrid.check_manifest(manifest(entries), found)
    with pytest.raises(HandoffError, match="missing"):
        hybrid.check_manifest(manifest(entries[:-1]), found)
    short = [e if e.name != "k" else wire.LayerExport(**{**vars(e), "shape": (6, 2, 8)}) for e in entries]
    with pytest.raises(HandoffError, match="expected"):
        hybrid.check_manifest(manifest(short), found)
    halved = [e if e.name != "ssm" else wire.LayerExport(**{**vars(e), "dtype": "bfloat16"}) for e in entries]
    with pytest.raises(HandoffError, match="float32"):
        hybrid.check_manifest(manifest(halved), found)
    with pytest.raises(HandoffError, match="single-rank"):
        hybrid.check_manifest(manifest(entries, tp_size=2), found)


def test_build_caches_puts_every_entry_where_the_model_reads_it():
    found = geometry()
    values, entries, chunks = entries_and_chunks(split=3)
    result = PrefillResult((manifest(entries),), tuple(chunks), 0, 0.0, 0.0)
    caches = hybrid.build_caches(mx, make_cache, found, result, TOKENS, mx.bfloat16)
    keys, vals, index_keys, positions = caches[2].state
    assert keys.shape == (1, 2, 7, 8) and mx.array_equal(keys[0], values[(2, "qsa", "k")].transpose(1, 0, 2))
    assert mx.array_equal(vals[0], values[(2, "qsa", "v")].transpose(1, 0, 2))
    assert mx.array_equal(index_keys[0], values[(2, "qsa", "index_keys")])
    assert positions.tolist() == [list(range(7))]
    for layer in (0, 1):
        assert mx.array_equal(caches[layer][0][0], values[(layer, "gdn", "conv")])
        assert caches[layer][1].dtype == mx.float32
        assert mx.array_equal(caches[layer][1][0], values[(layer, "gdn", "ssm")])
    assert mx.array_equal(caches[1][2][0], values[(1, "ple", "conv")])
    assert caches[1][3].tolist() == [TOKENS[-2:]] and caches[1][3].dtype == mx.int64


def test_build_caches_refuses_an_incomplete_entry():
    found = geometry()
    _, entries, chunks = entries_and_chunks(split=3)
    missing_frame = [c for c in chunks if not (c.layer.name == "k" and c.row_start == 3)]
    result = PrefillResult((manifest(entries),), tuple(missing_frame), 0, 0.0, 0.0)
    with pytest.raises(HandoffError, match="incomplete"):
        hybrid.build_caches(mx, make_cache, found, result, TOKENS, mx.bfloat16)


MTP_MODEL = SimpleNamespace(args=CONFIG, _omlx_mtp_decode_enabled=True, _omlx_mtp_chain=True,
                            get_mtp_module=lambda: object(), make_mtp_cache=lambda: [QSAKVCache()])


def mtp_entries_and_chunks(start=8, rows=6, layer=3):
    """The MTP head's entries as an MTP producer sends them: one row fewer than the prompt."""
    spec = {
        "k": ((rows, 2, 8), ["block", "head", "head_dim"]),
        "v": ((rows, 2, 8), ["block", "head", "head_dim"]),
        "index_keys": ((rows, 6), ["block", "index_dim"]),
        "hidden": ((1, 10), ["block", "hidden"]),
    }
    values, entries, chunks = {}, [], []
    for position, (name, (shape, dims)) in enumerate(spec.items(), start):
        array = (mx.arange(int(mx.prod(mx.array(shape)).item())) + 1000 * position).reshape(shape).astype(mx.bfloat16)
        values[name] = array
        entry = wire.LayerExport.from_dict({"index": position, "kind": "mtp", "layer": layer, "name": name,
                                            "shape": list(shape), "dims": dims, "dtype": "bfloat16"})
        entries.append(entry)
        chunks.append(Chunk(0, entry, 0, shape[0], array.view(mx.uint16).reshape(-1)))
    return values, entries, chunks


def test_mtp_entries_are_checked_only_when_the_model_drafts_with_its_head():
    _, entries, _ = entries_and_chunks()
    _, mtp, _ = mtp_entries_and_chunks()
    plain = geometry()
    assert plain.mtp_layer == -1
    hybrid.check_manifest(manifest(entries + mtp), plain)  # ignored: nothing here drafts with them
    found, reason = hybrid.hybrid_geometry(MTP_MODEL, make_cache())
    assert found is not None, reason
    assert found.mtp_layer == 3 and found.shapes[("mtp", "hidden")] == (1, 10)
    hybrid.check_manifest(manifest(entries + mtp), found)
    hybrid.check_manifest(manifest(entries), found)  # a producer without MTP sends none
    _, whole, _ = mtp_entries_and_chunks(rows=7)
    with pytest.raises(HandoffError, match="expected"):
        hybrid.check_manifest(manifest(entries + whole), found)
    with pytest.raises(HandoffError, match="MTP entries differ"):
        hybrid.check_manifest(manifest(entries + mtp[:-1]), found)


def test_build_mtp_gives_the_heads_history_and_the_pending_hidden_state():
    found, _ = hybrid.hybrid_geometry(MTP_MODEL, make_cache())
    values, entries, chunks = entries_and_chunks()
    mtp_values, mtp, mtp_chunks = mtp_entries_and_chunks()
    result = PrefillResult((manifest(entries + mtp),), tuple(chunks + mtp_chunks), 0, 0.0, 0.0)
    caches = hybrid.build_caches(mx, make_cache, found, result, TOKENS, mx.bfloat16)
    assert caches[2].state[0].shape == (1, 2, 7, 8)  # the backbone ignores the head's entries
    head, hidden = hybrid.build_mtp(mx, MTP_MODEL, found, result, len(TOKENS), mx.bfloat16)
    keys, vals, index_keys, positions = head[0].state
    assert keys.shape == (1, 2, 6, 8) and mx.array_equal(keys[0], mtp_values["k"].transpose(1, 0, 2))
    assert mx.array_equal(vals[0], mtp_values["v"].transpose(1, 0, 2))
    assert mx.array_equal(index_keys[0], mtp_values["index_keys"])
    assert positions.tolist() == [list(range(6))]
    assert hidden.shape == (1, 1, 10) and mx.array_equal(hidden[0], mtp_values["hidden"])
    plain = PrefillResult((manifest(entries),), tuple(chunks), 0, 0.0, 0.0)
    assert hybrid.build_mtp(mx, MTP_MODEL, found, plain, len(TOKENS), mx.bfloat16) is None


def test_the_handoffs_mtp_history_stands_in_for_the_prefix_sidecar():
    from omlx.remote_prefill.service import _HandedOffMtp

    calls = []
    base = SimpleNamespace(block_size=16, restore_mtp_prefix_snapshot=lambda *a, **k: calls.append(a) or "sidecar")
    snapshot = SimpleNamespace(boundary_tokens=7)
    view = _HandedOffMtp(base, snapshot)
    assert view.restore_mtp_prefix_snapshot(TOKENS, 7) is snapshot and not calls
    assert view.restore_mtp_prefix_snapshot(TOKENS, 5) == "sidecar" and view.block_size == 16
