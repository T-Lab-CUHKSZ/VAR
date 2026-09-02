"""Shared validation and agreement metrics for the blind audit."""
from __future__ import annotations

from typing import Iterable


JUDGES = ("harmbench", "llamaguard", "wildguard", "strongreject")


def binary(value, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes"}:
        return 1
    if normalized in {"0", "false", "no"}:
        return 0
    raise ValueError(f"{field} must be binary, found {value!r}")


def specificity(value) -> int:
    try:
        score = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"specificity must be an integer from 1 to 5, found {value!r}") from error
    if score not in range(1, 6):
        raise ValueError(f"specificity must be an integer from 1 to 5, found {value!r}")
    return score


def agreement_metrics(truth: Iterable[int], prediction: Iterable[int]) -> dict:
    truth = [binary(value, "truth") for value in truth]
    prediction = [binary(value, "prediction") for value in prediction]
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must be non-empty and have equal length")

    tp = sum(t == 1 and p == 1 for t, p in zip(truth, prediction))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, prediction))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, prediction))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, prediction))
    n = len(truth)
    observed = (tp + tn) / n
    truth_positive = (tp + fn) / n
    predicted_positive = (tp + fp) / n
    kappa_expected = (
        truth_positive * predicted_positive
        + (1 - truth_positive) * (1 - predicted_positive)
    )
    kappa = (
        (observed - kappa_expected) / (1 - kappa_expected)
        if kappa_expected < 1
        else float("nan")
    )
    mean_positive = (truth_positive + predicted_positive) / 2
    ac1_expected_disagreement = 2 * mean_positive * (1 - mean_positive)
    ac1 = (
        (observed - ac1_expected_disagreement) / (1 - ac1_expected_disagreement)
        if ac1_expected_disagreement < 1
        else float("nan")
    )
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "raw_agreement": observed,
        "cohen_kappa": kappa,
        "pabak": 2 * observed - 1,
        "gwet_ac1": ac1,
    }
