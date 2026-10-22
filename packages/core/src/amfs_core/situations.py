"""Situation identity: which declared labels name one kind of task.

A run declares the kind of task it is in (``Run.begin(situation=...)``,
``retrieve(situation=...)``) as free text, and spells it differently from run
to run. The outcome record is partitioned by that label — the situation's own
record is what :func:`amfs_core.actions.recommend` plans from, and an empty
record is an ``abstain`` — so the spelling decides whether an agent is served
its own record or nothing. Measured on the regime-change study (Beam 1.0,
2026-10-21, 8 cells): one class of ticket was declared under 14 distinct
labels per cell on average (``csv export timeout``, ``csv export timeout
web``, ``data export failure web``, ``web data export timeout large date
range`` …); compared as folded strings, the situation's record was non-empty
on 37 % of the first searches after the change, and the recommendation
abstained on 70 %. Under the rule below the record was non-empty on 93 %.

This module is the rule, in three layers:

* :func:`fold` — case, whitespace and punctuation folded, noise words that
  say what a label is *of* rather than what it is about dropped (``ticket``,
  ``issue``, ``failure``), the words compared as a set (``declined card`` is
  ``card declined``). This is the key every record is stored and compared
  under, unchanged from before identity existed.
* :func:`same_situation` — the token rule: two folded labels name one
  situation when one's stemmed tokens are all in the other's with at least
  :data:`MIN_SHARED` in common (a qualifier was added: ``csv export
  timeout`` ⊂ ``csv export timeout web``), or the Jaccard of the two sets is
  at least :data:`SIMILARITY_FLOOR`. Stemming is light (``pinned`` / ``pins``
  / ``pin``; ``dependencies`` / ``dependency``) and both sides go through it.
  The floors are precision-first, because the record this partitions is
  planned from: a wrong join serves another class's winner as ``act`` (the
  ops-queue incident of 2026-09-22), a missed join is the abstain the agent
  had anyway. A swapped token never joins — ``jest snapshot · UI PR`` and
  ``jest snapshot · backend-only PR`` share three tokens of four and stay
  apart, where a Jaccard floor of 0.5 with partial containment (the hosted
  canary router's rule) joins them. Measured on 24 cells of the regime
  studies (three models, 2026-10-22): purity 1.000 with no mixed cluster;
  the situation's record non-empty on 81 % of the first searches after a
  change (exact labels: 55 %); the hosted rule reached 94 % with one mixed
  cluster in 351.
* :func:`canonicalize` — the record's labels, oldest first, each joined to
  the first canonical it matches (:func:`nearest`) or opening a new one; and
  :func:`resolve`, a declared label joined to the record's canonicals the
  same way. First spelling wins, so the identity is stable as the record
  grows: a new label never re-homes an old one.

The token rule misses synonyms with no stem in common (``data export error``
for ``csv export timeout``; ``double charge`` for ``duplicate billing``). A
caller with an embedder or an alias table supplies a *resolver* — ``(folded
label, canonicals) -> canonical | None`` — consulted only when the token
rule finds nothing, so the second stage can only join, never split. Nothing
here is stemmed across classes on purpose: ``ios`` and ``android``, ``ui`` and
``backend`` differ by one token and must stay apart, and the floors are chosen
so that a swapped token never joins two labels — only an added one does.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

#: Jaccard over stemmed tokens at or above which two labels are one situation.
#: High enough that one swapped token out of four (``ui`` for ``backend``)
#: never reaches it: 3/5 = 0.6.
SIMILARITY_FLOOR = 0.75
#: Share of the shorter label's tokens that must appear in the longer one for
#: the containment rule: all of them (``csv export timeout`` ⊂ ``csv export
#: timeout web``). A partial containment joins a swap.
CONTAINMENT_FLOOR = 1.0
#: Tokens two labels must share before the containment rule applies at all,
#: so a one-token label does not join every label that mentions the token.
MIN_SHARED = 2

#: Words that carry no class information in a situation label: what the
#: label is *of* ("ticket", "issue"), not what it is about. Dropped before
#: labels are compared, so ``card declined ticket`` and ``card declined`` are
#: one record.
NOISE_WORDS = frozenset({
    "a", "an", "the", "of", "on", "in", "for", "with", "and", "or", "to", "at", "by",
    "ticket", "tickets", "issue", "issues", "problem", "problems", "case", "cases",
    "request", "requests", "task", "tasks", "failure", "failures", "error", "errors",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ing", "ed", "es", "s")

#: ``(folded label, canonicals) -> canonical or None``: a second stage a
#: caller supplies (alias table, embedder), asked only when the token rule
#: finds nothing.
Resolver = Callable[[str, Sequence[str]], "str | None"]


def fold(value: Any, covariates: Mapping[str, Any] | None = None) -> str:
    """A situation label as stored and compared.

    Case, whitespace and punctuation are folded (``Card-Declined`` is ``card
    declined``), noise words are dropped (``card declined ticket``), and the
    words are compared as a set (``declined card``). Nothing is stemmed here:
    the fold is the record's key, and two labels one stem apart are still two
    keys — :func:`same_situation` is where they meet. A label made only of
    noise words is compared as typed. ``""`` for anything that is not a
    non-empty string.

    With *covariates* — what the run declared about the task alongside the
    label (``plan=team``, ``device=Android``) — their keys and values are
    dropped from the label too: ``android card declined plan team`` is
    ``card declined``. A covariate is, by declaration, a property of the
    instance and not of the kind, so it cannot be part of the kind's
    identity; and an agent writing labels puts them there — on the
    2026-10-21 regime study Beam's 20 card-declined tickets on one cell
    carried 17 spellings, 12 of them one situation plus the ticket's plan or
    platform. A label that is nothing *but* covariates keeps them: it still
    has to be keyed.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    text = "".join(ch if ch.isalnum() else " " for ch in value.casefold())
    words = [w for w in text.split() if w not in NOISE_WORDS]
    if not words:
        words = text.split()
    if isinstance(covariates, Mapping) and covariates:
        drop = _covariate_tokens(covariates)
        kept = [w for w in words if w not in drop]
        if kept:
            words = kept
    return " ".join(sorted(set(words)))


def _covariate_tokens(covariates: Mapping[str, Any]) -> frozenset[str]:
    """The tokens a run's covariates would leave in a label: each key and
    each scalar value, folded as the label is."""
    out: set[str] = set()
    for key, value in covariates.items():
        for part in (key, value):
            if isinstance(part, bool) or part is None:
                continue
            if isinstance(part, str | int | float):
                out.update(_TOKEN_RE.findall(str(part).casefold()))
    return frozenset(out)


def stem(token: str) -> str:
    """A light suffix stripper, enough for the plural and participle forms an
    agent varies between run and run: ``pinned`` / ``pins`` / ``pin`` →
    ``pin``, ``dependencies`` → ``dependency``, ``declined`` / ``decline`` →
    ``declin``, ``timeouts`` → ``timeout``. Both sides of every comparison go
    through it, so the stems need only agree, not be words. Never shortens a
    token under three letters."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) >= 4 and token[-1] == token[-2] and token[-1] not in "aeiou":
        token = token[:-1]
    if len(token) >= 4 and token.endswith("e"):
        token = token[:-1]
    return token


def tokens(folded: str) -> frozenset[str]:
    """The stemmed token set of a folded label."""
    return frozenset(stem(t) for t in _TOKEN_RE.findall(folded))


def similarity(a: frozenset[str], b: frozenset[str]) -> tuple[float, float, int]:
    """``(jaccard, containment, shared)`` of two token sets; containment is the
    share of the smaller set found in the larger."""
    if not a or not b:
        return 0.0, 0.0, 0
    shared = len(a & b)
    jaccard = shared / len(a | b)
    containment = shared / min(len(a), len(b))
    return jaccard, containment, shared


def same_situation(a: str, b: str) -> bool:
    """Whether two folded labels name one situation under the stated rule."""
    if a == b:
        return True
    jaccard, containment, shared = similarity(tokens(a), tokens(b))
    if jaccard >= SIMILARITY_FLOOR:
        return True
    return shared >= MIN_SHARED and containment >= CONTAINMENT_FLOOR


def nearest(
    folded: str, canonicals: Iterable[str], *, resolver: Resolver | None = None,
) -> str | None:
    """The canonical label *folded* belongs to, or ``None`` when it names a
    new situation. The token rule decides first: ties go to the higher
    Jaccard, then containment, then the earlier canonical (iteration order).
    When it finds nothing and a *resolver* is given, the resolver's answer
    stands if it names one of the canonicals."""
    best: tuple[float, float, int] | None = None
    chosen: str | None = None
    mine = tokens(folded)
    canonicals = list(canonicals)
    for canonical in canonicals:
        if canonical == folded:
            return canonical
        jaccard, containment, shared = similarity(mine, tokens(canonical))
        if not (jaccard >= SIMILARITY_FLOOR
                or (shared >= MIN_SHARED and containment >= CONTAINMENT_FLOOR)):
            continue
        score = (jaccard, containment, shared)
        if best is None or score > best:
            best, chosen = score, canonical
    if chosen is not None or resolver is None or not canonicals:
        return chosen
    try:
        answer = resolver(folded, canonicals)
    except Exception:  # noqa: BLE001 - the second stage is best-effort
        return None
    return answer if isinstance(answer, str) and answer in canonicals else None


def canonicalize(
    labels: Iterable[str], *, resolver: Resolver | None = None,
) -> dict[str, str]:
    """Each distinct folded label in *labels* → its canonical, the labels
    taken in the order given (oldest first, so the first spelling of a
    situation is its canonical and a later spelling never re-homes an earlier
    one). Empty labels are skipped."""
    canonicals: list[str] = []
    out: dict[str, str] = {}
    for label in labels:
        if not label or label in out:
            continue
        mine = tokens(label)
        matches = [c for c in canonicals if same_situation(label, c)]
        if matches and all(mine < tokens(c) for c in matches):
            # The base spelling arrived after a qualified form (``csv export
            # timeout`` after ``csv export timeout web`` and ``csv export
            # timeout android``, which share it but not each other): it is
            # the situation they are of, it becomes the canonical, and the
            # qualified forms join under it — so a later sibling (``csv
            # export timeout ios``) finds the base and not one cousin. A join
            # only ever grows a record, so the identity stays stable for
            # every reader of the earlier canonicals.
            for c in matches:
                canonicals.remove(c)
                for spelling, canonical in out.items():
                    if canonical == c:
                        out[spelling] = label
            canonicals.append(label)
            out[label] = label
            continue
        found = nearest(label, canonicals, resolver=resolver)
        if found is None:
            canonicals.append(label)
            found = label
        out[label] = found
    return out


def resolve(
    folded: str, canonical_of: Mapping[str, str], *, resolver: Resolver | None = None,
) -> str:
    """The canonical a declared *folded* label belongs to under the record's
    identity (*canonical_of*, from :func:`canonicalize`): its own canonical
    when the record holds the spelling, else the nearest canonical, else the
    label itself (a situation the record has not seen)."""
    if not folded:
        return ""
    if folded in canonical_of:
        return canonical_of[folded]
    canonicals = list(dict.fromkeys(canonical_of.values()))
    return nearest(folded, canonicals, resolver=resolver) or folded


__all__ = [
    "CONTAINMENT_FLOOR",
    "MIN_SHARED",
    "NOISE_WORDS",
    "Resolver",
    "SIMILARITY_FLOOR",
    "canonicalize",
    "fold",
    "nearest",
    "resolve",
    "same_situation",
    "similarity",
    "stem",
    "tokens",
]
