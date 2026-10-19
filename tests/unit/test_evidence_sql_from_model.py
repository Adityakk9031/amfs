"""t2-11e: the SQL evidence functions are rendered from the Python model.

Migration 012 is a generated file. If it and ``render_evidence_sql()`` differ,
a constant or table changed on one side only — the exact drift migrations 008
and 009 made possible by carrying the arithmetic twice.
"""

from __future__ import annotations

from pathlib import Path

from amfs_core import evidence as ev
from amfs_core.models import OUTCOME_MULTIPLIERS
from amfs_postgres import evidence_sql

MIGRATIONS = Path(evidence_sql.__file__).parent / "migrations"


def test_migration_012_is_the_rendering_of_the_python_model() -> None:
    committed = (MIGRATIONS / evidence_sql.MIGRATION_NAME).read_text(encoding="utf-8")
    assert committed == evidence_sql.render_evidence_sql(), (
        "migration 012 differs from amfs_core.evidence; regenerate with "
        "`python -m amfs_postgres.evidence_sql > .../migrations/012_evidence_from_model.sql`"
    )


def test_the_rendering_carries_every_constant_and_table_of_the_model() -> None:
    sql = evidence_sql.render_evidence_sql()
    assert f"c_decay CONSTANT NUMERIC := {ev.EVIDENCE_DECAY!r}" in sql
    assert f"c_prior_strength CONSTANT NUMERIC := {float(ev.PRIOR_STRENGTH)!r}" in sql
    assert f"c_discredit CONSTANT NUMERIC := {ev.DISCREDIT_THRESHOLD!r}" in sql
    assert f"c_first_strike_wins CONSTANT INTEGER := {ev.FIRST_STRIKE_MIN_WINS}" in sql
    assert f"c_max_validators CONSTANT INTEGER := {ev.MAX_VALIDATORS}" in sql
    for name, sev in ev.SEVERITY.items():
        assert f"WHEN '{name}' THEN {float(sev)!r}" in sql
    for key, mult in OUTCOME_MULTIPLIERS.items():
        name = key.value if hasattr(key, "value") else str(key)
        assert f"WHEN '{name}' THEN {float(mult)!r}" in sql
    for name in ev.SUCCESS_TYPES:
        assert f"'{name}'" in sql.split("amfs_outcome_is_known")[0]
    # No literal of the old hand-written bodies survives outside the constants:
    # the step reads the named constants, not 0.8 / 2.0 / 0.5 / 3 / 10 inline.
    step = sql.split("RETURNS INTEGER AS $$", 1)[1]
    body = step.split("BEGIN", 1)[1]
    for literal in ("* 0.8", "(2.0 * prior", "< 0.5", ">= 0.5", ">= 3 THEN", "> 10 THEN", "- 10"):
        assert literal not in body, literal


def test_the_rendering_is_the_same_functions_008_and_009_defined() -> None:
    """012 replaces, it does not add: the same four scalar functions and the
    eight-argument step, so the 009 trigger keeps calling what it called."""
    sql = evidence_sql.render_evidence_sql()
    for fn in (
        "amfs_outcome_is_success(t TEXT)",
        "amfs_outcome_is_known(t TEXT)",
        "amfs_outcome_severity(t TEXT)",
        "amfs_outcome_multiplier(t TEXT)",
    ):
        assert f"CREATE OR REPLACE FUNCTION {fn}" in sql
    assert sql.count("CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(") == 1
    nine = (MIGRATIONS / "009_action_priors.sql").read_text(encoding="utf-8")
    sig_009 = nine.split("CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(", 1)[1].split(
        ") RETURNS", 1
    )[0]
    sig_012 = sql.split("CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(", 1)[1].split(
        ") RETURNS", 1
    )[0]
    assert sig_012 == sig_009
    assert "CREATE OR REPLACE FUNCTION amfs_propagate_outcome" not in sql
