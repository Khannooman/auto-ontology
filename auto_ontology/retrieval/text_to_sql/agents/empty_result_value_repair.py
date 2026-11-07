# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empty-result value repair, and the last-resort retry behind it.

Runs right after execution, only when the query came back empty. It checks the
filter literals against the columns' actual distinct values and, when a literal
is a near-miss of a real value (wrong case / spelling), routes once to the
existing reconstruction node with a targeted hint. On the common path (non-empty
result) it is a single boolean check — no probes, no LLM call.

This is the last node before the response, so when the result is still empty and
neither this check nor the empty-LIKE one found anything to fix, it sends the
query back for one undiagnosed regeneration rather than returning nothing. The
measured causes of a surviving empty result are not literal typos: a filter
naming a column that doesn't hold that value, a join dropping the rows, or a
date compared in the wrong format. None are repairable from a probe, and all are
things a fresh derivation can get right — so unlike the targeted paths above,
this one deliberately omits ``error_known_fixable`` and lets reconstruction's
classifier reach for different tables if it judges the data to be elsewhere.

An empty result can legitimately be the correct answer, so the retry is
one-shot and never rejects the regenerated query for also coming back empty —
the cost of being wrong here is one extra reconstruction, not a lost answer.
"""

from __future__ import annotations

from typing import Any, Dict

import sqlglot
from sqlglot import exp

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    _get_sql_code,
    _is_empty_db_result,
    _parse_db_rows,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.retrieval.text_to_sql.db_probe.literal_check import (
    build_value_repair_error,
    find_literal_mismatches,
    find_numeric_scale_mismatches,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState


def build_empty_result_retry_error() -> str:
    """Instruction for an empty result no check could diagnose.

    Deliberately names the causes that actually produce one (wrong column for
    the value, over-restrictive join, format mismatch) instead of asking for a
    generic retry, and asks for a re-derivation rather than an edit — a patch of
    the existing SQL tends to preserve the very table choice that is wrong.
    """
    return (
        "SQL executed successfully but returned an empty result set. The query "
        "is valid and its filter literals were checked against the database, so "
        "this is not a typo in a value — the query is asking the wrong question "
        "of the data. Re-derive it from the schema rather than editing it, and "
        "reconsider in particular:\n"
        "- whether each filtered value is stored in the column being filtered, "
        "or actually lives in a different column or table;\n"
        "- whether a join is dropping the rows of interest (a row that exists "
        "in one table may have no counterpart in another);\n"
        "- whether a date, time, or number is compared in the format and units "
        "the column really uses.\n"
        "If the question can be answered from different tables, prefer those."
    )


def _outer_select(sql: str) -> exp.Select | None:
    """The statement itself when it is one ``SELECT``, otherwise nothing.

    A parse failure or a ``UNION`` stays out of the null-aggregate path.
    Those shapes are left to the existing empty-result check.
    """
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001 — unparsable SQL is not a null aggregate
        return None
    return tree if isinstance(tree, exp.Select) else None


def _is_aggregate_expression(expression: exp.Expression) -> bool:
    node = expression
    while isinstance(node, (exp.Alias, exp.Paren)):
        node = node.this
    return isinstance(node, exp.AggFunc)


def _is_scalar_null_aggregate(sql: str, db_result: Any) -> bool:
    """One-row scalar aggregate whose every value is JSON null.

    ``SUM`` over no input rows returns one row of NULL, so the empty-result
    check never sees it and the literal probe never runs. ``COUNT`` of no
    rows and ``SUM`` of real zeros are numeric zeros, and a selected NULL
    column is not an aggregate, so neither is included. A grouped aggregate
    can legitimately be NULL for a group that exists.
    """
    rows = _parse_db_rows(db_result)
    if len(rows) != 1:
        return False
    values = list(rows[0].values())
    if not values or any(value is not None for value in values):
        return False
    select = _outer_select(sql or "")
    if select is None or select.args.get("group") is not None:
        return False
    expressions = select.expressions or []
    return bool(expressions) and all(
        _is_aggregate_expression(expression) for expression in expressions
    )


class EmptyResultValueRepairAgent(BaseAgent):
    """Repair a bad filter literal, or regenerate an empty result outright."""

    def __init__(self) -> None:
        super().__init__("empty_result_value_repair")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        if path_state.get("sql_response_from_db") is None:
            self.logger.warning("No SQL execution result found for value repair")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)
        db_result = path_state.get("sql_response_from_db")

        empty = _is_empty_db_result(db_result)
        # A scalar SUM/AVG/MIN/MAX over no rows is one JSON-null row, not an
        # empty list. Probe it the same way. A zero, a count of zero, or a
        # null selected from a real row stays a result.
        null_aggregate = not empty and _is_scalar_null_aggregate(sql_code, db_result)

        # Common path: query returned rows → nothing to repair, zero cost.
        if not empty and not null_aggregate:
            return {"decision": "valid_sql", "path_state": path_state}

        mismatches: list[dict[str, Any]] = []
        if path_state.get("value_repair_attempted"):
            self.logger.info("Value repair already attempted — skipping the probe")
        else:
            connectors = state.get("connectors") or []
            relevant_tables = list(path_state.get("relevant_tables") or [])
            connector = resolve_connector_from_tables(relevant_tables, connectors)
            dialect = getattr(connector, "dialect", None)

            with ProbeExecutor(connector) as executor:
                # Reached only when the query came back empty, so the extra
                # high-cardinality literal check is warranted here (and only here).
                mismatches = find_literal_mismatches(
                    executor, dialect, sql_code, result_was_empty=True
                )
                mismatches += find_numeric_scale_mismatches(executor, dialect, sql_code)

        if mismatches:
            path_state["value_repair_attempted"] = True
            path_state["error"] = build_value_repair_error(mismatches)
            # This error was derived from a live probe against the query's own
            # already-joined tables — the fix is always "use the real value we
            # just found there," never "go search for a new table." Skip
            # reconstruction's LLM error-classification for it (see
            # sql_reconstruction.py) so it can't be misread as missing_data.
            path_state["error_known_fixable"] = True
            self.logger.info(
                "[%s] Empty result — routing to reconstruction to fix %d literal(s): %s",
                path_state.get("task_id", "?"),
                len(mismatches),
                [f"{m['table']}.{m['column']}='{m['used']}'" for m in mismatches],
            )
            return {"decision": "invalid_sql", "path_state": path_state}

        if null_aggregate:
            # The probe saw no case or whitespace variant. The NULL may be the
            # stored measure, so it is not rewritten to zero and not sent
            # through the undiagnosed regeneration used for an empty list.
            self.logger.info(
                "[%s] Scalar aggregate is NULL and no stored string variant "
                "matched the filter — leaving the NULL in place",
                path_state.get("task_id", "?"),
            )
            return {"decision": "valid_sql", "path_state": path_state}

        if path_state.get("empty_result_retry_attempted"):
            self.logger.info(
                "Empty result persisted after an undiagnosed retry — passing through"
            )
            return {"decision": "valid_sql", "path_state": path_state}

        # Nothing diagnosable, and an empty result answers nothing. Regenerate
        # once, without error_known_fixable so the classifier may look elsewhere.
        path_state["empty_result_retry_attempted"] = True
        path_state["error"] = build_empty_result_retry_error()
        self.logger.info(
            "[%s] Empty result with no diagnosable cause — routing to "
            "reconstruction for one undiagnosed retry",
            path_state.get("task_id", "?"),
        )
        return {"decision": "invalid_sql", "path_state": path_state}


__all__ = ["EmptyResultValueRepairAgent", "build_empty_result_retry_error"]
