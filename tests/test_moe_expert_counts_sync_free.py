# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The sync-free MoE routing arithmetic, on CPU, with no accelerator anywhere.

Two things replaced a blocking call on the grouped-GEMM path and both are pure
tensor algebra, so the whole contract is checkable on a bare runner:

1. ``count_tokens_per_expert`` stands in for
   ``torch.bincount(flat, minlength=E).to(dtype)``. "Sync-free" is a property of
   the CUDA kernel and cannot be observed here; what CAN be observed, and is what
   a wrong rewrite breaks, is that the two agree bit for bit on every routing
   shape, including the ones bincount handles by a special case (empty input, a
   single token, every token on one expert, an expert index never used, E=1).

2. the bias expansion in ``forward_native_grouped_mm`` gathers with
   ``index_select`` over the sorted expert ids instead of
   ``repeat_interleave(counts)``. Those are equal only because ``argsort`` is
   stable and the counts come from the same tensor, so the identity is asserted
   directly rather than assumed.

CPU only and GPU free by construction: no ``torch.cuda`` call appears below.
"""

import contextlib
import importlib.util
import os
import pathlib
import sys

import pytest
import torch


_DISABLE_GPU_INIT = "UNSLOTH_ZOO_DISABLE_GPU_INIT"


@contextlib.contextmanager
def _gpu_init_skipped():
    """``unsloth_zoo.__init__`` raises NotImplementedError when no accelerator is
    visible, and its non-skipped path also demands an importable ``unsloth`` plus
    ``UNSLOTH_IS_PRESENT``. The flag is read at import time only, so it is put back
    afterwards and never set when the package is already loaded."""
    previous = os.environ.get(_DISABLE_GPU_INIT)
    if "unsloth_zoo" not in sys.modules:
        os.environ[_DISABLE_GPU_INIT] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_DISABLE_GPU_INIT, None)
        else:
            os.environ[_DISABLE_GPU_INIT] = previous


def _load_moe_utils():
    with _gpu_init_skipped():
        try:
            from unsloth_zoo.temporary_patches import moe_utils

            return moe_utils
        except Exception:
            # ``temporary_patches/__init__`` imports every patch module and those need
            # transformers; moe_utils itself imports only torch and unsloth_zoo.mlx, so
            # load it from its file. It has no relative imports, so no parent package
            # stub is needed, and it is deliberately not registered in sys.modules:
            # nothing else in the session imports this name.
            import unsloth_zoo

            path = pathlib.Path(unsloth_zoo.__file__).parent / "temporary_patches" / "moe_utils.py"
            spec = importlib.util.spec_from_file_location("unsloth_zoo_moe_utils_under_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module


moe_utils = _load_moe_utils()
count_tokens_per_expert = moe_utils.count_tokens_per_expert


def _bincount_reference(flat_experts, num_experts, dtype = torch.int32):
    """Exactly the expression the patch replaced."""
    return torch.bincount(flat_experts.reshape(-1).long(), minlength = num_experts).to(dtype)


def _routing_cases():
    generator = torch.Generator().manual_seed(0)
    return [
        ("random", torch.randint(0, 8, (257,), generator = generator, dtype = torch.int64), 8),
        ("random_wide", torch.randint(0, 128, (4096,), generator = generator, dtype = torch.int64), 128),
        ("all_on_expert_0", torch.zeros(97, dtype = torch.int64), 8),
        ("all_on_last_expert", torch.full((97,), 7, dtype = torch.int64), 8),
        # Highest used index below E-1: bincount's own output would be short of
        # num_experts without minlength, so this is where a naive rewrite diverges.
        ("tail_experts_unused", torch.randint(0, 3, (64,), generator = generator, dtype = torch.int64), 16),
        ("empty", torch.zeros(0, dtype = torch.int64), 8),
        ("single_token", torch.tensor([3], dtype = torch.int64), 8),
        ("single_expert", torch.zeros(16, dtype = torch.int64), 1),
        ("two_dimensional", torch.randint(0, 6, (32, 4), generator = generator, dtype = torch.int64), 6),
        ("three_dimensional", torch.randint(0, 5, (4, 8, 2), generator = generator, dtype = torch.int64), 5),
    ]


_CASES = _routing_cases()
_CASE_IDS = [name for name, _, _ in _CASES]


@pytest.mark.parametrize("name,flat_experts,num_experts", _CASES, ids = _CASE_IDS)
def test_counts_are_bit_exact_against_bincount(name, flat_experts, num_experts):
    got = count_tokens_per_expert(flat_experts, num_experts, torch.int32)
    expected = _bincount_reference(flat_experts, num_experts, torch.int32)
    assert torch.equal(got, expected), (name, got, expected)
    assert got.shape == (num_experts,)
    assert got.dtype == torch.int32
    # Independent of bincount: every routed slot is counted exactly once.
    assert int(got.sum().item()) == flat_experts.numel()


@pytest.mark.parametrize("name,flat_experts,num_experts", _CASES, ids = _CASE_IDS)
def test_input_is_not_consumed(name, flat_experts, num_experts):
    before = flat_experts.clone()
    count_tokens_per_expert(flat_experts, num_experts, torch.int32)
    assert torch.equal(flat_experts, before)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.int16, torch.float32])
def test_requested_dtype_shape_and_device_are_honoured(dtype):
    flat_experts = torch.randint(0, 6, (128,), generator = torch.Generator().manual_seed(1))
    got = count_tokens_per_expert(flat_experts, 6, dtype)
    assert got.dtype == dtype
    assert got.shape == (6,)
    assert got.device == flat_experts.device
    assert torch.equal(got, _bincount_reference(flat_experts, 6, dtype))


def test_default_dtype_is_int32():
    """The callers pass torch.int32 explicitly; the default has to agree with them
    because `offs=` on the grouped GEMM is built from this tensor."""
    got = count_tokens_per_expert(torch.tensor([0, 1, 1, 3]), 4)
    assert got.dtype == torch.int32


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.int16, torch.uint8])
def test_narrow_integer_index_dtypes(index_dtype):
    """Router topk gives int64, but a caller that already narrowed its indices must
    not get a silent failure out of scatter_add_, which only accepts int64."""
    flat_experts = torch.randint(0, 5, (96,), generator = torch.Generator().manual_seed(2)).to(index_dtype)
    got = count_tokens_per_expert(flat_experts, 5, torch.int32)
    assert torch.equal(got, _bincount_reference(flat_experts, 5, torch.int32))
    assert flat_experts.dtype == index_dtype


def test_counts_hold_under_deterministic_algorithms():
    """scatter_add_ accumulates with atomics, so the docstring's claim that it stays
    usable under torch.use_deterministic_algorithms(True) is worth pinning."""
    flat_experts = torch.randint(0, 16, (512,), generator = torch.Generator().manual_seed(3))
    expected = _bincount_reference(flat_experts, 16, torch.int32)
    was_enabled = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        got = count_tokens_per_expert(flat_experts, 16, torch.int32)
    finally:
        torch.use_deterministic_algorithms(was_enabled)
    assert torch.equal(got, expected)


def _sorted_routing(flat_top_k):
    """The exact three lines forward_native_grouped_mm runs before the bias add."""
    sorted_indices = torch.argsort(flat_top_k, stable = True)
    return sorted_indices, flat_top_k[sorted_indices]


@pytest.mark.parametrize(
    "num_experts,top_k,num_tokens,seed",
    [
        (8, 2, 37, 0),
        (4, 1, 16, 1),
        (16, 4, 9, 2),
        (2, 2, 64, 3),
    ],
)
def test_index_select_bias_expansion_equals_repeat_interleave(num_experts, top_k, num_tokens, seed):
    generator = torch.Generator().manual_seed(seed)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k), generator = generator)
    flat_top_k = top_k_index.view(-1)
    counts = count_tokens_per_expert(flat_top_k, num_experts, torch.int32)
    sorted_indices, sorted_expert_ids = _sorted_routing(flat_top_k)

    # The identity holds only if the permutation really is grouped by expert.
    assert torch.equal(sorted_expert_ids, sorted_expert_ids.sort().values)

    bias = torch.arange(num_experts * 5, dtype = torch.float32).reshape(num_experts, 5)
    gathered = bias.index_select(0, sorted_expert_ids)
    repeated = bias.repeat_interleave(counts, dim = 0)
    assert torch.equal(gathered, repeated)
    assert gathered.shape == (num_tokens * top_k, 5)


def test_bias_expansion_when_some_experts_receive_nothing():
    """Zero-count experts are the case repeat_interleave handles by emitting no rows;
    index_select matches only because those ids never appear in the sorted ids."""
    num_experts = 6
    flat_top_k = torch.tensor([4, 0, 4, 0, 0, 4, 1], dtype = torch.int64)
    counts = count_tokens_per_expert(flat_top_k, num_experts, torch.int32)
    assert torch.equal(counts, torch.tensor([3, 1, 0, 0, 3, 0], dtype = torch.int32))

    _, sorted_expert_ids = _sorted_routing(flat_top_k)
    bias = torch.arange(num_experts, dtype = torch.float32).unsqueeze(1)
    assert torch.equal(bias.index_select(0, sorted_expert_ids), bias.repeat_interleave(counts, dim = 0))


def test_bias_expansion_row_order_follows_the_permuted_inputs():
    """The bias row added to permuted row i has to be the bias of the expert that row
    i was routed to; a transposed or unsorted gather would still have the right shape."""
    num_experts, top_k = 5, 2
    top_k_index = torch.randint(0, num_experts, (13, top_k), generator = torch.Generator().manual_seed(4))
    flat_top_k = top_k_index.view(-1)
    sorted_indices, sorted_expert_ids = _sorted_routing(flat_top_k)
    bias = torch.arange(num_experts, dtype = torch.float32).unsqueeze(1)
    gathered = bias.index_select(0, sorted_expert_ids)
    for row, position in enumerate(sorted_indices.tolist()):
        assert gathered[row, 0].item() == float(flat_top_k[position].item())


def test_get_routing_indices_agrees_with_bincount_and_groups_by_expert():
    num_experts = 12
    selected_experts = torch.randint(0, num_experts, (48, 3), generator = torch.Generator().manual_seed(5))
    counts, gather_indices = moe_utils._get_routing_indices(selected_experts, num_experts)
    flat = selected_experts.view(-1)
    assert torch.equal(counts, _bincount_reference(flat, num_experts, torch.int32))
    assert counts.dtype == torch.int32
    grouped = flat[gather_indices]
    assert torch.equal(grouped, grouped.sort().values)
    # stable=True: within one expert the original order is preserved.
    for expert in range(num_experts):
        positions = gather_indices[grouped == expert]
        assert torch.equal(positions, positions.sort().values)


def test_debug_counter_is_off_by_default():
    """The counter wrapper is a Dynamo side effect that re-traced the compiled MoE
    block every step, so the default build must expose the bare function."""
    debug_on = os.environ.get("UNSLOTH_MOE_COUNT_DEBUG", "0") == "1"
    assert moe_utils._COUNT_DEBUG is debug_on
    before = moe_utils._EXPERT_COUNT_CALLS[0]
    count_tokens_per_expert(torch.tensor([0, 1, 1]), 2, torch.int32)
    after = moe_utils._EXPERT_COUNT_CALLS[0]
    if debug_on:
        assert moe_utils.count_tokens_per_expert is not moe_utils._count_tokens_per_expert
        assert after == before + 1
    else:
        assert moe_utils.count_tokens_per_expert is moe_utils._count_tokens_per_expert
        assert after == before


def test_docstring_survives_the_debug_wrapper():
    assert "WITHOUT synchronising" in (count_tokens_per_expert.__doc__ or "")
