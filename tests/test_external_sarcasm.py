"""Tests for the external sarcasm-data helpers (experiments/data/). Run: uv run pytest -q tests/test_external_sarcasm.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "data"))
from ext_common import has_s_tag, strip_s_tag  # noqa: E402


@pytest.mark.parametrize("text, expected", [
    ("Small price to pay for freedom!!! /s", "Small price to pay for freedom!!!"),
    ("Great idea (/s)", "Great idea"),
    ("Oh aye, very clever /S.", "Oh aye, very clever"),
    ("Brilliant. /sarcasm", "Brilliant."),
    ("nice one ^(/s)", "nice one"),
    ("Line one\n\n/s - if it isn't obvious.", "Line one"),
    ("Sure, because trains run on time here (/s obviously)", "Sure, because trains run on time here"),
])
def test_closing_tag_is_stripped(text, expected):
    assert strip_s_tag(text) == expected


@pytest.mark.parametrize("text", [
    "You forgot the /s",          # talks about the tag
    "Is this /s?",
    "The adding of an /s should be a criminal offence",  # mid-sentence
    "Totally /s agree",
    "> quoted /s\nreply here",    # tag only inside a quote
    "wind 10 m/s today",          # not a tag
    "see r/s and /sub",
])
def test_non_tag_uses_are_rejected(text):
    assert strip_s_tag(text) is None


def test_has_s_tag_is_broad_for_negative_filtering():
    assert has_s_tag("Totally /s agree")
    assert not has_s_tag("10 m/s")


def test_dedupe_matcher():
    pytest.importorskip("ahocorasick")
    from dedupe import Matcher, keep_mask, norm_list

    ref = "The quick brown fox jumps over the lazy dog near the river bank today"
    m = Matcher(norm_list([ref, "ok"]))
    keep = keep_mask([
        "PREFIX: " + ref.upper() + "!!!",  # contains a reference (>= 40 chars)
        ref[4:60],                         # contained in a reference
        "ok",                              # exact short match
        "o k",                             # normalises to 'o k', not 'ok'
        "something unrelated that is certainly longer than forty characters",
    ], m)
    assert keep.tolist() == [False, False, False, True, True]
