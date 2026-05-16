"""Robustness coverage for the fused lm_head + cross_entropy installer.

Complements `tests/test_fused_forward_install.py` with edge cases the
rewriter and adapter must handle (or correctly refuse): wrappers around
the lm_head call on the inference branch, decorator preservation,
non-canonical model shapes (Gemma3 softcap, CSM auxiliary loss, Bloom
explicit kwargs), eligibility gating (ForConditionalGeneration, composite
heads, transformers version floor), and pre-shifted-labels scaling.
"""

from __future__ import annotations

import ast
import linecache

import pytest


# ---------------------------------------------------------------------------
# AST rewriter -- forward-shape fixtures
# ---------------------------------------------------------------------------


KW_CANONICAL_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""

FLOAT_WRAPPER_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    lm_logits = self.lm_head(hidden_states).float()
    loss = None
    if labels is not None:
        loss = self.loss_function(lm_logits, labels, self.config.vocab_size, **kwargs)
    return (loss, lm_logits)
"""

GEMMA_SOFTCAP_SRC = """
def forward(self, labels=None, **kwargs):
    hidden_states = self.model(**kwargs).last_hidden_state
    logits = self.lm_head(hidden_states)
    if self.config.final_logit_softcapping is not None:
        logits = logits / self.config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.final_logit_softcapping
    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
    return (loss, logits)
"""

CSM_MULTI_STMT_SRC = """
def forward(self, labels=None, **kwargs):
    backbone_outputs = self.backbone_model(**kwargs)
    hidden_states = backbone_outputs[0]
    backbone_logits = self.lm_head(hidden_states)
    loss = None
    backbone_loss = None
    if labels is not None:
        backbone_labels = labels[:, :, 0]
        backbone_loss = self.loss_function(logits=backbone_logits, labels=backbone_labels, vocab_size=self.config.vocab_size, **kwargs)
        loss = backbone_loss + 0
    return (loss, backbone_logits)
"""

BLOOM_EXPLICIT_KWARGS_SRC = """
def forward(self, labels=None, **kwargs):
    transformer_outputs = self.transformer(**kwargs)
    hidden_states = transformer_outputs[0]
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, num_items_in_batch=kwargs.get("num_items_in_batch"))
    return (loss, logits)
"""

ORELSE_SRC = """
def forward(self, labels=None, **kwargs):
    hidden_states = self.model(**kwargs).last_hidden_state
    logits = self.lm_head(hidden_states)
    if labels is not None:
        loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)
    else:
        loss = self.aux_zero()
    return (loss, logits)
"""

NESTED_GUARD_SRC = """
def forward(self, labels=None, use_cache=False, **kwargs):
    hidden_states = self.model(**kwargs).last_hidden_state
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        if not use_cache:
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)
    return (loss, logits)
"""

DECORATED_SRC = """
@can_return_tuple
@auto_docstring
def forward(self, labels=None, **kwargs):
    hidden_states = self.model(**kwargs).last_hidden_state
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)
    return (loss, logits)
"""

ALIASED_LABELS_SRC = """
def forward(self, labels=None, **kwargs):
    hidden_states = self.model(**kwargs).last_hidden_state
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        shifted = labels.clone()
        loss = self.loss_function(logits, shifted, self.config.vocab_size, **kwargs)
    return (loss, logits)
"""

COMPOSITE_HEAD_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.cls(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""


def _rewrite(src):
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    return rewrite_forward_source(src)


def test_else_branch_preserves_float_wrapper():
    new_src, cap = _rewrite(FLOAT_WRAPPER_SRC)
    assert new_src is not None and cap is not None
    tree = ast.parse(new_src)
    fn = tree.body[0]
    if_node = next(n for n in fn.body if isinstance(n, ast.If))
    else_assigns = [s for s in if_node.orelse if isinstance(s, ast.Assign)]
    lm_head_assign = next(
        s for s in else_assigns
        if isinstance(s.targets[0], ast.Name) and s.targets[0].id == cap.logits_name
    )
    assert ".float()" in ast.unparse(lm_head_assign.value)


def test_else_branch_preserves_slice_wrapper():
    src = KW_CANONICAL_SRC.replace(
        "logits = self.lm_head(hidden_states)",
        "logits = self.lm_head(hidden_states[:, slice(-1, None), :])",
    )
    new_src, _ = _rewrite(src)
    assert new_src is not None
    assert "hidden_states[:, slice(-1, None), :]" in new_src


def test_gemma_softcap_block_rejects_rewrite():
    new_src, cap = _rewrite(GEMMA_SOFTCAP_SRC)
    assert new_src is None
    assert cap is None


def test_csm_multi_statement_labels_block_rejected():
    new_src, cap = _rewrite(CSM_MULTI_STMT_SRC)
    assert new_src is None
    assert cap is None


def test_orelse_on_labels_if_rejected():
    new_src, cap = _rewrite(ORELSE_SRC)
    assert new_src is None
    assert cap is None


def test_nested_guard_in_labels_block_rejected():
    new_src, cap = _rewrite(NESTED_GUARD_SRC)
    assert new_src is None
    assert cap is None


def test_aliased_labels_arg_rejected():
    new_src, cap = _rewrite(ALIASED_LABELS_SRC)
    assert new_src is None
    assert cap is None


def test_bloom_explicit_num_items_kwarg_forwarded():
    new_src, cap = _rewrite(BLOOM_EXPLICIT_KWARGS_SRC)
    assert new_src is not None and cap is not None
    assert "num_items_in_batch=" in new_src
    assert (
        "kwargs.get('num_items_in_batch')" in new_src
        or 'kwargs.get("num_items_in_batch")' in new_src
    )


def test_can_return_tuple_decorator_preserved_but_auto_docstring_dropped():
    new_src, _ = _rewrite(DECORATED_SRC)
    assert new_src is not None
    assert "@can_return_tuple" in new_src
    assert "@auto_docstring" not in new_src


# ---------------------------------------------------------------------------
# install_for_class / install_for_module eligibility
# ---------------------------------------------------------------------------


_SYNTH_COUNTER = 0


def _synthetic_class(src: str, name: str):
    global _SYNTH_COUNTER
    _SYNTH_COUNTER += 1
    path = f"<unsloth-robustness-synth-{_SYNTH_COUNTER}.py>"
    text = src.lstrip("\n")
    linecache.cache[path] = (
        len(text), None, [line + "\n" for line in text.splitlines()], path,
    )
    ns = {}
    exec(compile(text, path, "exec"), ns)
    cls = type(name, (), {"forward": ns["forward"]})
    cls.__module__ = "transformers.models.synthetic.modeling_synthetic"
    return cls


@pytest.fixture
def fresh_registry():
    from unsloth_zoo.fused_losses import forward_install as fi
    with fi._REGISTRY_LOCK:
        fi._PATCHED.clear()
        fi._UNMATCHED.clear()
        fi._FAILED.clear()
        fi._CANONICAL_FORWARDS.clear()
    yield fi
    with fi._REGISTRY_LOCK:
        fi._PATCHED.clear()
        fi._UNMATCHED.clear()
        fi._FAILED.clear()
        fi._CANONICAL_FORWARDS.clear()


@pytest.fixture
def env_on(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_FORWARD", "1")


def test_for_conditional_generation_not_installed(fresh_registry, env_on):
    cls = _synthetic_class(KW_CANONICAL_SRC, name="SyntheticForConditionalGeneration")
    original = cls.forward
    assert fresh_registry.install_for_class(cls) is False
    assert cls.forward is original


def test_composite_head_attr_not_installed(fresh_registry, env_on):
    cls = _synthetic_class(COMPOSITE_HEAD_SRC, name="SyntheticForCausalLM")
    original = cls.forward
    assert fresh_registry.install_for_class(cls) is False
    assert cls.forward is original
    assert cls.__qualname__ in fresh_registry._UNMATCHED
    assert "non-linear-head" in fresh_registry._UNMATCHED[cls.__qualname__]


def test_install_for_class_respects_version_floor(fresh_registry, env_on, monkeypatch):
    monkeypatch.setattr(fresh_registry, "_transformers_version_ok", lambda: False)
    cls = _synthetic_class(KW_CANONICAL_SRC, name="SyntheticForCausalLM")
    original = cls.forward
    assert fresh_registry.install_for_class(cls) is False
    assert cls.forward is original


def test_install_for_module_respects_version_floor(fresh_registry, env_on, monkeypatch):
    import types
    monkeypatch.setattr(fresh_registry, "_transformers_version_ok", lambda: False)
    cls = _synthetic_class(KW_CANONICAL_SRC, name="SyntheticForCausalLM")
    mod = types.ModuleType("transformers.models.synthetic.modeling_synthetic")
    mod.SyntheticForCausalLM = cls
    assert fresh_registry.install_for_module(mod) == 0


# ---------------------------------------------------------------------------
# Adapter shift_labels fallback + gradient-accumulation scaling
# ---------------------------------------------------------------------------


def _make_lm_head_and_hidden(B=2, T=4, H=8, V=16):
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    lm_head = nn.Linear(H, V, bias=False)
    hidden = torch.randn(B, T, H)
    labels = torch.randint(0, V, (B, T))
    return lm_head, hidden, labels, V


def _shift(labels):
    shifted = labels.clone()
    shifted[..., :-1] = labels[..., 1:]
    shifted[..., -1] = -100
    return shifted


def test_pre_shifted_tensor_fallback_divides_by_int_n_items():
    import torch
    from unsloth_zoo.fused_losses.forward_adapter import unsloth_fused_lm_head_loss

    lm_head, hidden, labels, V = _make_lm_head_and_hidden()
    shifted = _shift(labels)
    n = 2

    ref_logits = torch.nn.functional.linear(hidden, lm_head.weight, None)
    expected = torch.nn.functional.cross_entropy(
        ref_logits.view(-1, V).float(),
        shifted.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / n

    out = unsloth_fused_lm_head_loss(
        hidden, lm_head, labels,
        shift_labels=shifted, num_items_in_batch=n,
    )
    assert torch.allclose(out.detach(), expected.detach(), atol=1e-5)


def test_pre_shifted_tensor_fallback_mean_when_no_n_items():
    import torch
    from unsloth_zoo.fused_losses.forward_adapter import unsloth_fused_lm_head_loss

    lm_head, hidden, labels, V = _make_lm_head_and_hidden()
    shifted = _shift(labels)

    ref_logits = torch.nn.functional.linear(hidden, lm_head.weight, None)
    expected = torch.nn.functional.cross_entropy(
        ref_logits.view(-1, V).float(),
        shifted.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    out = unsloth_fused_lm_head_loss(hidden, lm_head, labels, shift_labels=shifted)
    assert torch.allclose(out.detach(), expected.detach(), atol=1e-5)


def test_shift_labels_false_bool_uses_labels_unshifted():
    import torch
    from unsloth_zoo.fused_losses.forward_adapter import unsloth_fused_lm_head_loss

    lm_head, hidden, labels, V = _make_lm_head_and_hidden()
    shifted = _shift(labels)

    ref_logits = torch.nn.functional.linear(hidden, lm_head.weight, None)
    expected = torch.nn.functional.cross_entropy(
        ref_logits.view(-1, V).float(),
        shifted.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    out = unsloth_fused_lm_head_loss(
        hidden, lm_head, shifted, shift_labels=False,
    )
    assert torch.allclose(out.detach(), expected.detach(), atol=1e-5)


def test_pre_shifted_tensor_fallback_divides_by_tensor_n_items():
    import torch
    from unsloth_zoo.fused_losses.forward_adapter import unsloth_fused_lm_head_loss

    lm_head, hidden, labels, V = _make_lm_head_and_hidden()
    shifted = _shift(labels)
    n = torch.tensor(3.0)

    ref_logits = torch.nn.functional.linear(hidden, lm_head.weight, None)
    expected = torch.nn.functional.cross_entropy(
        ref_logits.view(-1, V).float(),
        shifted.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / 3.0

    out = unsloth_fused_lm_head_loss(
        hidden, lm_head, labels,
        shift_labels=shifted, num_items_in_batch=n,
    )
    assert torch.allclose(out.detach(), expected.detach(), atol=1e-5)


def test_int_n_items_divisor_promotes_to_tensor():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("UnslothFusedLoss.forward requires a CUDA device")
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss

    B, T, H, V = 1, 8, 8, 16
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, V, (B, T), device="cuda")

    loss = unsloth_fused_ce_loss(
        trainer=None,
        hidden_states=hidden,
        lm_head_weight=weight,
        lm_head_bias=None,
        labels=labels,
        n_items=3,
        torch_compile=False,
    )
    assert torch.isfinite(loss)
