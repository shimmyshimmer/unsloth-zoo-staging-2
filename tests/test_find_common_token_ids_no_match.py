# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""``_find_common_token_ids`` when the matched core is not found in the component.

The helper locates the common core inside the component's full tokenization to
recover the optional left/right tokens around it:

    for j in range(len(original)):
        if original[j : j + len(substring)] == substring: break
    optional_left  = original[:j]
    optional_right = original[j+len(substring):]

``substring`` is the common core across probe variants that each carry an appended
``[0]`` sentinel, so it is not guaranteed to be a sublist of ``original``. When the
loop found nothing it fell through with ``j`` at the last index and sliced anyway,
returning optional tokens that were never matched; when ``original`` was empty ``j``
was never bound at all (``UnboundLocalError``).

CPU-pure and offline: the tokenizer is a local stub, no weights are loaded.
"""

import pytest

from unsloth_zoo.dataset_utils import _find_common_token_ids, train_on_responses_only


class StubTokenizer:
    """Deterministic tokenizer: each character maps to its ordinal unless overridden."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def __call__(self, text, add_special_tokens=False):
        class _Result:
            pass

        result = _Result()
        result.input_ids = (
            list(self.mapping[text]) if text in self.mapping else [ord(c) for c in text]
        )
        return result


# Probe variants share only the appended [0] sentinel, so the common core comes back
# as [0], which is absent from the plain tokenization of "X" ([1, 2]).
NO_COMMON_CORE = {
    "X": [1, 2],
    "\nX": [9, 1, 2],
    "\nX\n": [9, 1, 2, 9],
    "X\n": [1, 2, 9],
    "\n\nX": [9, 9, 1, 2],
}


def test_core_absent_from_component_reports_no_optional_context():
    """The regression: when the core is not a sublist of the component's tokens,
    no optional left/right context may be invented."""
    tokenizer = StubTokenizer(NO_COMMON_CORE)

    substring, optional_left, optional_right = _find_common_token_ids("X", tokenizer, False)

    original = tokenizer("X").input_ids
    matched = any(
        original[j : j + len(substring)] == substring for j in range(len(original))
    )
    assert not matched, "fixture no longer exercises the no-match path; update it"
    assert optional_left == [] and optional_right == [], (
        f"core {substring} is absent from {original} yet the helper returned "
        f"optional_left={optional_left} optional_right={optional_right}, which were "
        "never matched"
    )


def test_component_tokenizing_to_nothing_does_not_raise_unbound_local():
    """An empty tokenization must not blow up on an unbound loop variable."""
    tokenizer = StubTokenizer()

    try:
        substring, optional_left, optional_right = _find_common_token_ids("", tokenizer, True)
    except UnboundLocalError as exception:  # pragma: no cover - the bug being fixed
        pytest.fail(f"empty component raised UnboundLocalError: {exception}")

    assert optional_left == [] and optional_right == []


def test_empty_explicit_marker_raises_named_error():
    """An explicitly-passed empty marker must fail with a named error, not IndexError
    on A_must[0]. Only reachable via explicit args - auto-detect rejects empty
    markers before returning."""
    tokenizer = StubTokenizer()

    with pytest.raises(ValueError, match="response_part tokenizes to no tokens"):
        train_on_responses_only(
            None, "user:", "", tokenizer=tokenizer, return_function=True
        )

    with pytest.raises(ValueError, match="instruction_part tokenizes to no tokens"):
        train_on_responses_only(
            None, "", "assistant:", tokenizer=tokenizer, return_function=True
        )


def test_normal_marker_still_recovers_optional_context():
    """Guard the happy path: when the core IS found, the surrounding tokens are still
    returned as optional left/right context."""
    tokenizer = StubTokenizer()

    substring, optional_left, optional_right = _find_common_token_ids("\nAB\n", tokenizer, False)

    original = tokenizer("\nAB\n").input_ids
    assert substring, "expected a non-empty core for a normal marker"
    where = next(
        j for j in range(len(original)) if original[j : j + len(substring)] == substring
    )
    assert optional_left == original[:where]
    assert optional_right == original[where + len(substring) :]


def test_force_match_marker_matches_at_index_zero():
    """force_match=True tokenizes the component verbatim, so the core is the whole
    tokenization and there is no optional context on either side."""
    tokenizer = StubTokenizer()

    substring, optional_left, optional_right = _find_common_token_ids("AB", tokenizer, True)

    assert substring == tokenizer("AB").input_ids
    assert optional_left == [] and optional_right == []
