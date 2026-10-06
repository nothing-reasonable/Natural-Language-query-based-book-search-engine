from __future__ import annotations

import pytest

from web import _parse_options


def test_rerank_can_be_toggled_without_diagnostics() -> None:
    assert _parse_options({"rerank": True}, diagnostics=False).rerank is True
    assert _parse_options({"rerank": False}, diagnostics=False).rerank is False


def test_diagnostic_only_search_options_remain_hidden() -> None:
    options = _parse_options(
        {"plan_mode": "always", "rag_fusion": True, "trace": True, "trace_rerank": True},
        diagnostics=False,
    )

    assert options.plan_mode is None
    assert options.rag_fusion is None
    assert options.trace is False
    assert options.trace_rerank is False


def test_rerank_toggle_requires_a_boolean() -> None:
    with pytest.raises(ValueError, match="rerank must be a boolean"):
        _parse_options({"rerank": "false"}, diagnostics=False)
