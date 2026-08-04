# SIM C (T5) -- cross-platform end-to-end proof, designed to run on a bare CI runner.
#
# The PR's own tests use a hand-written stub tokenizer. That is fine for pinning the contract,
# but it cannot show that the fix works against a REAL tokenizer on Windows and macOS, where
# the multiprocessing start method is `spawn` and `sft_prepare_dataset` takes a different
# num_proc branch than it does on Linux.
#
# So build a real `PreTrainedTokenizerFast` in-process -- no Hub download, no weights -- with
# the two properties that make the bug reachable:
#   * a multi-character BOS token, and
#   * a chat template that emits BOS through the Jinja VARIABLE `{{ bos_token }}`, so the
#     literal never appears in the template source and the second detector arm stays dead.
# That is exactly the Llama-3 / Gemma-3 / Mistral shape measured on the real models.
#
# Then assert the user-visible outcome: text produced by apply_chat_template(tokenize=False)
# must tokenize to EXACTLY ONE leading BOS id.

import multiprocessing
import platform

import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from unsloth_zoo.dataset_utils import sft_prepare_dataset

BOS = "<|begin_of_text|>"
EOS = "<|eot_id|>"
# BOS is emitted as a Jinja variable, never as a literal -- this is what kills detector arm B.
JINJA_BOS_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{%- endfor %}"
)


def build_real_tokenizer():
    vocab = {BOS: 0, EOS: 1, "user": 2, "assistant": 3, ":": 4, "hi": 5, "yo": 6, "[UNK]": 7}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token=BOS, eos_token=EOS, unk_token="[UNK]",
    )
    tok.chat_template = JINJA_BOS_TEMPLATE
    # A real post-processor: prepend BOS whenever add_special_tokens=True. This is the
    # behaviour that turns "text already has BOS" into a doubled BOS.
    from tokenizers.processors import TemplateProcessing
    tok._tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS} $A", special_tokens=[(BOS, 0)],
    )
    return tok


class Args:
    def __init__(self, num_proc=None):
        self.max_length = 64
        self.dataset_text_field = "text"
        self.remove_unused_columns = True
        if num_proc is not None:
            self.dataset_num_proc = num_proc


class DummyTrainer:
    def __init__(self):
        self.model = None
        self.data_collator = None


def leading_bos(ids, bos_id):
    n = 0
    for t in ids:
        if t != bos_id:
            break
        n += 1
    return n


def _prepare(dataset, tok, num_proc=None):
    prepared = sft_prepare_dataset(
        DummyTrainer(), dataset, tok, Args(num_proc),
        packing=False, formatting_func=None, dataset_name="train")
    return [list(r["input_ids"]) for r in prepared]


def test_platform_banner():
    print(f"\nplatform={platform.system()} python={platform.python_version()} "
          f"start_method={multiprocessing.get_start_method()}")
    assert True


@pytest.mark.parametrize("num_proc", [None, 1])
def test_chat_rendered_text_gets_exactly_one_bos(num_proc):
    """The regression, against a REAL tokenizer, on whatever platform this runs on."""
    tok = build_real_tokenizer()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        tokenize=False)
    assert text.startswith(BOS), "template must emit BOS for this test to mean anything"
    assert BOS not in (tok.chat_template or ""), "BOS must NOT appear literally in the template"

    rows = _prepare(Dataset.from_dict({"text": [text, text]}), tok, num_proc)
    for ids in rows:
        assert leading_bos(ids, tok.bos_token_id) == 1, (
            f"expected exactly 1 leading BOS on {platform.system()} "
            f"(start_method={multiprocessing.get_start_method()}, num_proc={num_proc}), "
            f"got {leading_bos(ids, tok.bos_token_id)} in {ids}")


@pytest.mark.parametrize("num_proc", [None, 1])
def test_plain_text_still_gets_its_bos(num_proc):
    """The other direction: text WITHOUT a leading BOS must still receive exactly one."""
    tok = build_real_tokenizer()
    rows = _prepare(Dataset.from_dict({"text": ["user: hi", "user: yo"]}), tok, num_proc)
    for ids in rows:
        assert leading_bos(ids, tok.bos_token_id) == 1, ids


def test_empty_first_row_does_not_crash():
    """Baseline raises IndexError here (it indexed character 0 of an empty string).
    The PR handles it. Pinning the incidental fix so it cannot silently regress."""
    tok = build_real_tokenizer()
    rows = _prepare(Dataset.from_dict({"text": ["", "user: hi"]}), tok, 1)
    assert len(rows) == 2
