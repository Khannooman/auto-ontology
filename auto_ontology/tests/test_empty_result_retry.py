# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the undiagnosed-empty-result retry in EmptyResultValueRepairAgent.

The targeted literal repair only covers a value that is a near-miss of a real
one. Everything else that leaves a result empty — a filter on a column that
doesn't hold the value, an over-restrictive join, a format mismatch — reaches
the end of the graph undiagnosed, and is regenerated once instead of being
returned as an empty answer.
"""

import pandas as pd

from auto_ontology.retrieval.text_to_sql.agents.empty_result_value_repair import (
    EmptyResultValueRepairAgent,
)


def _state(db_result, **path_state):
    """Minimal state: no connectors, so every probe fails and finds no mismatch."""
    return {
        "connectors": [],
        "path_state": {
            "sql_code": "SELECT * FROM t WHERE a = 'x' AND b = 'y'",
            "sql_response_from_db": db_result,
            "relevant_tables": [],
            **path_state,
        },
    }


def test_non_empty_result_passes_through_untouched():
    """The common path must stay free: no probe, no retry, no state churn."""
    out = EmptyResultValueRepairAgent().execute(_state([{"a": 1}]))

    assert out["decision"] == "valid_sql"
    assert "empty_result_retry_attempted" not in out["path_state"]


def test_undiagnosed_empty_result_is_regenerated_once():
    """Nothing to repair, but an empty result answers nothing — go back once."""
    out = EmptyResultValueRepairAgent().execute(_state([]))

    assert out["decision"] == "invalid_sql"
    assert out["path_state"]["empty_result_retry_attempted"] is True
    assert "empty result set" in out["path_state"]["error"]


def test_undiagnosed_retry_lets_reconstruction_classify_the_error():
    """Unlike the literal repair, this has no probe evidence behind it, so
    reconstruction must stay free to decide the data lives elsewhere and search
    for other tables (see sql_reconstruction's missing_data path).
    """
    out = EmptyResultValueRepairAgent().execute(_state([]))

    assert "error_known_fixable" not in out["path_state"]


def test_retry_happens_at_most_once():
    """A second empty result is accepted rather than looped on."""
    out = EmptyResultValueRepairAgent().execute(
        _state([], empty_result_retry_attempted=True)
    )

    assert out["decision"] == "valid_sql"


class _Frames:
    """Connector returning scripted frames. ``execute`` records that it ran."""

    dialect = "postgres"

    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self.frames = list(frames)
        self.calls = 0

    def execute(self, sql: str, timeout_s: float | None = None) -> pd.DataFrame:
        self.calls += 1
        return self.frames.pop(0)


class _MustNotProbe:
    dialect = "postgres"

    def execute(self, sql: str, timeout_s: float | None = None) -> pd.DataFrame:
        raise AssertionError(sql)


def _aggregate_state(sql: str, db_result, connector):
    return {
        "connectors": [connector],
        "path_state": {
            "sql_code": sql,
            "sql_response_from_db": db_result,
            "relevant_tables": [],
        },
    }


def _many_values() -> pd.DataFrame:
    return pd.DataFrame({"value": [f"value-{i}" for i in range(21)]})


def test_null_sum_with_trailing_whitespace_variant_is_repaired():
    """SUM over no rows is one NULL, and a whitespace variant is a real value."""
    connector = _Frames(
        [
            _many_values(),
            pd.DataFrame({"hit": []}),
            pd.DataFrame({"value": ["flash drive "]}),
        ]
    )
    sql = "SELECT SUM(amount) AS total FROM items WHERE label = 'flash drive'"
    out = EmptyResultValueRepairAgent().execute(
        _aggregate_state(sql, ['[{"total": null}]'], connector)
    )

    assert out["decision"] == "invalid_sql"
    assert out["path_state"]["error_known_fixable"] is True
    assert "LOWER(TRIM(label::text))" in out["path_state"]["error"]
    assert "empty_result_retry_attempted" not in out["path_state"]


def test_null_sum_without_a_stored_variant_stays_null():
    """No case or whitespace sibling means the NULL is left unchanged."""
    connector = _Frames(
        [
            _many_values(),
            pd.DataFrame({"hit": []}),
            pd.DataFrame({"value": []}),
        ]
    )
    sql = "SELECT SUM(amount) AS total FROM items WHERE label = 'absent'"
    out = EmptyResultValueRepairAgent().execute(
        _aggregate_state(sql, [{"total": None}], connector)
    )

    assert out["decision"] == "valid_sql"
    assert "error" not in out["path_state"]
    assert "empty_result_retry_attempted" not in out["path_state"]


def test_exact_literal_with_null_measure_is_not_rewritten():
    """The filter matches a stored value, so a NULL measure is real data."""
    connector = _Frames(
        [
            _many_values(),
            pd.DataFrame({"hit": [1]}),
        ]
    )
    sql = "SELECT SUM(amount) AS total FROM items WHERE label = 'present'"
    out = EmptyResultValueRepairAgent().execute(
        _aggregate_state(sql, [{"total": None}], connector)
    )

    assert out["decision"] == "valid_sql"
    assert connector.calls == 2


def test_zero_sum_and_zero_count_are_not_probed():
    """A numeric zero, including COUNT of no rows, is an answer."""
    connector = _MustNotProbe()
    agent = EmptyResultValueRepairAgent()
    summed = agent.execute(
        _aggregate_state(
            "SELECT SUM(amount) AS total FROM items WHERE label = 'present'",
            [{"total": 0}],
            connector,
        )
    )
    counted = agent.execute(
        _aggregate_state(
            "SELECT COUNT(*) AS n FROM items WHERE label = 'absent'",
            ['[{"n": 0}]'],
            connector,
        )
    )

    assert summed["decision"] == "valid_sql"
    assert counted["decision"] == "valid_sql"


def test_selected_null_and_grouped_null_are_not_probed():
    """A NULL column or a NULL group is not an empty scalar aggregate."""
    connector = _MustNotProbe()
    agent = EmptyResultValueRepairAgent()
    selected = agent.execute(
        _aggregate_state(
            "SELECT note FROM items WHERE label = 'present'",
            [{"note": None}],
            connector,
        )
    )
    grouped = agent.execute(
        _aggregate_state(
            "SELECT SUM(amount) AS total FROM items GROUP BY label",
            [{"total": None}],
            connector,
        )
    )

    assert selected["decision"] == "valid_sql"
    assert grouped["decision"] == "valid_sql"


def test_retry_still_runs_after_a_spent_literal_repair():
    """The proactive check may already have used up the literal repair earlier
    in the run. That says nothing about this empty result, so the undiagnosed
    retry is still owed.
    """
    out = EmptyResultValueRepairAgent().execute(_state([], value_repair_attempted=True))

    assert out["decision"] == "invalid_sql"
    assert out["path_state"]["empty_result_retry_attempted"] is True
