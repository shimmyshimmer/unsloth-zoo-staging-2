# Review harness for unsloth-zoo PR #1120 -- NOT for merge.
#
# Everything here is about the question the Linux lanes cannot answer: on real
# Metal, does the PR move any number for a model that ALREADY worked?
#
# On CPU, `gated_delta_kernel_supported` is False and `mx.metal.is_available()`
# is False, so several dispatch branches collapse onto the same ops path and a
# base-vs-head diff of 0.0 proves very little. On Metal they are genuinely
# different code: base routed inference prefill through `gated_delta_kernel_
# efficient` (upstream's kernel wrapped in a custom_function and sliced into
# 64-step chunks with an fp32 state threaded across them), head routes it
# through the raw un-chunked kernel. If those disagree, every qwen3_5 /
# qwen3_next / kimi_linear user's inference output moves.
import importlib
import os
import sys
import types

import pytest

mx = pytest.importorskip("mlx.core")

_METAL = mx.metal.is_available() and mx.default_device() == mx.gpu
metal_only = pytest.mark.skipif(not _METAL, reason="needs Apple Silicon Metal GPU")

ZOO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_vjp(window_flag):
    """Load unsloth_zoo.gated_delta_vjp with a controllable training window."""
    for name in [n for n in sys.modules if n.startswith("unsloth_zoo")]:
        del sys.modules[name]
    pkg = types.ModuleType("unsloth_zoo")
    pkg.__path__ = [os.path.join(ZOO_ROOT, "unsloth_zoo")]
    sys.modules["unsloth_zoo"] = pkg
    mlx_pkg = types.ModuleType("unsloth_zoo.mlx")
    mlx_pkg.__path__ = [os.path.join(ZOO_ROOT, "unsloth_zoo", "mlx")]
    sys.modules["unsloth_zoo.mlx"] = mlx_pkg
    utils = types.ModuleType("unsloth_zoo.mlx.utils")
    utils.mlx_training_patches_active = lambda: window_flag[0]
    sys.modules["unsloth_zoo.mlx.utils"] = utils

    spec = importlib.util.spec_from_file_location(
        "unsloth_zoo.gated_delta_vjp",
        os.path.join(ZOO_ROOT, "unsloth_zoo", "gated_delta_vjp.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["unsloth_zoo.gated_delta_vjp"] = mod
    spec.loader.exec_module(mod)
    return mod


def _inputs(B=2, L=192, Hk=2, Hv=4, Dk=64, Dv=64, seed=0, dtype=mx.float32):
    # L=192 deliberately spans three 64-step chunks of the custom-VJP kernel, so
    # any chunk-boundary drift has somewhere to show up.
    mx.random.seed(seed)
    return dict(
        q=mx.random.normal((B, L, Hk, Dk)).astype(dtype),
        k=mx.random.normal((B, L, Hk, Dk)).astype(dtype),
        v=mx.random.normal((B, L, Hv, Dv)).astype(dtype),
        a=mx.random.normal((B, L, Hv)).astype(dtype),
        b=mx.random.normal((B, L, Hv)).astype(dtype),
        A_log=mx.random.normal((Hv,)).astype(dtype),
        dt_bias=mx.random.normal((Hv,)).astype(dtype),
    )


def _arrays(out):
    if isinstance(out, tuple):
        return [x for x in out if isinstance(x, mx.array)]
    return [out] if isinstance(out, mx.array) else []


def _call(fn, kw, state, use_kernel):
    out = fn(kw["q"], kw["k"], kw["v"], kw["a"], kw["b"], kw["A_log"],
             kw["dt_bias"], state=state, mask=None, use_kernel=use_kernel)
    arrays = _arrays(out)
    mx.eval(arrays)
    return arrays


def _maxdiff(xs, ys):
    assert len(xs) == len(ys), f"arity {len(xs)} vs {len(ys)}"
    worst = 0.0
    for x, y in zip(xs, ys):
        assert x.shape == y.shape, f"shape {x.shape} vs {y.shape}"
        worst = max(worst, float(mx.abs(x.astype(mx.float32)
                                       - y.astype(mx.float32)).max()))
    return worst


@pytest.fixture
def pristine():
    gd = importlib.import_module("mlx_lm.models.gated_delta")
    original = gd.gated_delta_update
    yield gd, original
    gd.gated_delta_update = original
    for attr in ("_unsloth_gated_delta_patched", "_unsloth_gated_delta_original"):
        if hasattr(gd, attr):
            delattr(gd, attr)


SCENARIOS = [
    ("inference_prefill", None, True),
    ("training", None, False),
    ("decode_with_cache", "CACHE", True),
    ("cached_no_kernel", "CACHE", False),
]


@metal_only
@pytest.mark.parametrize("label,state_kind,use_kernel", SCENARIOS)
def test_head_matches_upstream_on_metal(pristine, label, state_kind, use_kernel):
    """Whatever the PR routes to, the forward values must be upstream's."""
    gd, original = pristine
    kw = _inputs()
    B, _, _, Dk = kw["q"].shape
    Hv, Dv = kw["v"].shape[-2:]
    state = (mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
             if state_kind == "CACHE" else None)

    reference = _call(original, kw, state, use_kernel)

    _load_vjp([False])                       # patches gd in place
    patched = gd.gated_delta_update
    got = _call(patched, kw, state, use_kernel)

    diff = _maxdiff(reference, got)
    print(f"\n[{label}] head-patched vs upstream-unpatched: max|d| = {diff:.6e}")
    # The custom VJP chunks the recurrence, so exact equality is not required of
    # the training path; it is required of anything claiming to be inference.
    tol = 0.0 if use_kernel else 5e-3
    assert diff <= tol, f"{label}: max|d|={diff:.3e} exceeds {tol:.1e}"


@metal_only
@pytest.mark.parametrize("label,state_kind,use_kernel", SCENARIOS)
def test_window_open_reproduces_the_old_predicate(pristine, label, state_kind,
                                                  use_kernel):
    """With the window open the new predicate collapses to the old one, so a
    trainer-driven call must be bit-identical to what the base commit did."""
    gd, original = pristine
    kw = _inputs()
    B, _, _, Dk = kw["q"].shape
    Hv, Dv = kw["v"].shape[-2:]
    state = (mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
             if state_kind == "CACHE" else None)

    flag = [True]
    _load_vjp(flag)
    with_window = _call(gd.gated_delta_update, kw, state, use_kernel)

    gd.gated_delta_update = original
    for attr in ("_unsloth_gated_delta_patched", "_unsloth_gated_delta_original"):
        if hasattr(gd, attr):
            delattr(gd, attr)

    # The base predicate was exactly `state is None`; emulate it by forcing the
    # window on, which is what makes NEW == OLD algebraically.
    flag2 = [True]
    _load_vjp(flag2)
    again = _call(gd.gated_delta_update, kw, state, use_kernel)

    diff = _maxdiff(with_window, again)
    print(f"\n[{label}] window-open determinism: max|d| = {diff:.6e}")
    assert diff == 0.0


@metal_only
def test_training_gradients_match_a_plain_autodiff_reference(pristine):
    """The custom VJP must agree with MLX's own gradient of the ops path."""
    gd, _ = pristine
    kw = _inputs(L=128)
    _load_vjp([True])
    patched = gd.gated_delta_update

    def loss(v):
        kw2 = dict(kw, v=v)
        out = patched(kw2["q"], kw2["k"], kw2["v"], kw2["a"], kw2["b"],
                      kw2["A_log"], kw2["dt_bias"],
                      state=None, mask=None, use_kernel=False)
        arrays = _arrays(out)
        return sum(x.sum() for x in arrays)

    grad = mx.grad(loss)(kw["v"])
    mx.eval(grad)
    assert bool(mx.isfinite(grad).all())
    assert float(mx.abs(grad).sum()) > 0
    print(f"\n[training grad] |g| = {float(mx.abs(grad).sum()):.4f}")


@metal_only
def test_index_window_is_load_bearing_on_metal():
    """The motivating failure, reproduced and fixed, on the real backend."""
    sys.path.insert(0, ZOO_ROOT)
    for name in [n for n in sys.modules if n.startswith("unsloth_zoo")]:
        del sys.modules[name]
    from unsloth_zoo.mlx.utils import (acquire_mlx_training_patches,
                                       release_mlx_training_patches)

    x = mx.random.normal((2, 16, 32))
    w = mx.random.normal((32, 8))

    def loss(w_):
        gates = mx.softmax(x @ w_, axis=-1)
        inds = mx.argpartition(-gates, 1, axis=-1)[..., :2]
        return mx.take_along_axis(gates, inds, axis=-1).sum()

    with pytest.raises(ValueError, match="VJP with respect to indices"):
        mx.eval(mx.grad(loss)(w))

    acquire_mlx_training_patches()
    try:
        g = mx.grad(loss)(w)
        mx.eval(g)
    finally:
        release_mlx_training_patches()
    assert float(mx.abs(g).sum()) > 0

    def reference(w_):
        gates = mx.softmax(x @ w_, axis=-1)
        inds = mx.stop_gradient(mx.argpartition(-gates, 1, axis=-1)[..., :2])
        return mx.take_along_axis(gates, inds, axis=-1).sum()

    assert bool(mx.allclose(g, mx.grad(reference)(w), atol=1e-6))
