#!/usr/bin/env python3
"""Held-out-behavior register separability for paper Tables 9-10."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


REGISTERS = [
    "ABO_omegaverse", "character_study", "coffee_shop_au", "danmei_xianxia",
    "enemies_to_lovers", "epistolary", "frame_narrative", "hurt_comfort_nonabo",
    "screenplay_format", "second_person_pov", "slow_burn", "soulmate_au",
]
FEATURE_NAMES = [
    "mean_word_length",
    "mean_sentence_length",
    "comma_rate",
    "colon_rate",
    "quotation_rate",
    "first_person_pronoun_rate",
    "second_person_pronoun_rate",
    "third_person_pronoun_rate",
    "exclamation_rate",
    "question_rate",
    "capitalization_ratio",
    "newline_ratio",
    "digit_ratio",
    "type_token_ratio",
    "perplexity",
]
PRONOUNS = {
    "first": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"},
    "second": {"you", "your", "yours", "yourself", "yourselves"},
    "third": {
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    },
}


def structural_features(text: str, perplexity: float) -> np.ndarray:
    words = re.findall(r"\b\w+(?:[-'’]\w+)*\b", text)
    lower_words = [word.casefold() for word in words]
    sentences = [
        re.findall(r"\b\w+(?:[-'’]\w+)*\b", sentence)
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]
    char_count = max(len(text), 1)
    word_count = max(len(words), 1)
    alpha_count = max(sum(character.isalpha() for character in text), 1)
    mean_word_length = sum(len(word) for word in words) / word_count
    mean_sentence_length = (
        sum(len(sentence) for sentence in sentences) / len(sentences)
        if sentences
        else 0.0
    )
    values = [
        mean_word_length,
        mean_sentence_length,
        text.count(",") / char_count,
        text.count(":") / char_count,
        sum(text.count(mark) for mark in ('"', "“", "”")) / char_count,
        sum(word in PRONOUNS["first"] for word in lower_words) / word_count,
        sum(word in PRONOUNS["second"] for word in lower_words) / word_count,
        sum(word in PRONOUNS["third"] for word in lower_words) / word_count,
        text.count("!") / char_count,
        text.count("?") / char_count,
        sum(character.isupper() for character in text) / alpha_count,
        text.count("\n") / char_count,
        sum(character.isdigit() for character in text) / char_count,
        len(set(lower_words)) / word_count,
        float(perplexity),
    ]
    if len(values) != 15 or not np.isfinite(values).all():
        raise ValueError("structural feature vector must contain fifteen finite values")
    return np.asarray(values, dtype=float)


def load_perplexity(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result[record["prompt_id"]] = float(record["perplexity"])
    return result


def load_records(path: Path, perplexity_path: Path | None, strict_shape: bool) -> list[dict]:
    perplexities = load_perplexity(perplexity_path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        prompt_id = record.get("prompt_id")
        register = record.get("register")
        if register is None:
            arm = str(record.get("arm", ""))
            register = arm[:-7] if arm.endswith("_single") else arm
        behavior = record.get("behavior_id", record.get("behavior_idx"))
        turns = record.get("turns") or []
        text = turns[0] if turns else record.get("prompt")
        perplexity = perplexities.get(prompt_id, record.get("perplexity"))
        if not prompt_id or register not in REGISTERS or behavior is None or not isinstance(text, str):
            raise ValueError(f"line {line_number}: invalid prompt/register/behavior/text contract")
        if perplexity is None:
            raise ValueError(
                f"{prompt_id}: perplexity is required in the record or through --perplexity"
            )
        records.append(
            {
                "prompt_id": prompt_id,
                "register": register,
                "behavior": str(behavior),
                "text": text,
                "perplexity": float(perplexity),
            }
        )
    if strict_shape:
        behavior_counts = Counter(record["behavior"] for record in records)
        if len(records) != 3480 or len(behavior_counts) != 290 or set(behavior_counts.values()) != {12}:
            raise ValueError("strict separability input must be 290 behaviors x 12 registers = 3,480 prompts")
        if Counter(record["register"] for record in records) != Counter({name: 290 for name in REGISTERS}):
            raise ValueError("each register must contain exactly 290 prompts")
    return records


def purity_score(labels: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(clusters):
        counts = Counter(labels[clusters == cluster])
        total += max(counts.values())
    return total / len(labels)


def evaluate(records: list[dict], seed: int = 42) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    behaviors = sorted({record["behavior"] for record in records})
    if len(behaviors) != 290:
        raise ValueError("the reported split requires exactly 290 behaviors")
    train_behaviors, test_behaviors = train_test_split(
        behaviors, train_size=203, test_size=87, random_state=seed, shuffle=True
    )
    train_set, test_set = set(train_behaviors), set(test_behaviors)
    train_indices = np.array([record["behavior"] in train_set for record in records])
    test_indices = np.array([record["behavior"] in test_set for record in records])
    if np.any(train_indices & test_indices) or train_indices.sum() != 2436 or test_indices.sum() != 1044:
        raise AssertionError("behavior-group split must be disjoint and have 2,436/1,044 prompts")

    texts = [record["text"] for record in records]
    labels = np.asarray([record["register"] for record in records])
    structural = np.vstack(
        [structural_features(record["text"], record["perplexity"]) for record in records]
    )
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 4),
        n_features=6000,
        lowercase=True,
        alternate_sign=False,
        norm="l2",
    )
    character = vectorizer.transform(texts)

    classifiers = {
        "logistic": LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed),
        "centroid": NearestCentroid(),
        "knn15": KNeighborsClassifier(n_neighbors=15),
    }
    result = {
        "config": {
            "seed": seed,
            "train_behaviors": 203,
            "test_behaviors": 87,
            "train_prompts": 2436,
            "test_prompts": 1044,
            "structural_features": FEATURE_NAMES,
            "character_features": {
                "dimensions": 6000,
                "ngram_range": [3, 4],
                "normalization": "l2",
            },
        },
        "supervised": {"structural": {}, "character": {}},
        "clustering": {},
    }
    for feature_name, matrix in (("structural", structural), ("character", character)):
        for classifier_name, classifier in classifiers.items():
            if feature_name == "structural":
                estimator = make_pipeline(StandardScaler(), classifier)
            else:
                estimator = classifier
            estimator.fit(matrix[train_indices], labels[train_indices])
            prediction = estimator.predict(matrix[test_indices])
            result["supervised"][feature_name][classifier_name] = {
                "accuracy": float(accuracy_score(labels[test_indices], prediction)),
                "macro_f1": float(f1_score(labels[test_indices], prediction, average="macro")),
                "per_register_f1": {
                    register: float(value)
                    for register, value in zip(
                        REGISTERS,
                        f1_score(labels[test_indices], prediction, labels=REGISTERS, average=None),
                    )
                },
            }

        cluster_matrix = StandardScaler().fit_transform(matrix) if feature_name == "structural" else matrix
        clusters = KMeans(n_clusters=12, n_init=20, random_state=seed).fit_predict(cluster_matrix)
        result["clustering"][feature_name] = {
            "adjusted_rand_index": float(adjusted_rand_score(labels, clusters)),
            "purity": float(purity_score(labels, clusters)),
            "fit_scope": "all 3,480 unlabeled prompts",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--perplexity", help="JSONL with prompt_id and perplexity")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(
        Path(args.prompts),
        Path(args.perplexity) if args.perplexity else None,
        True,
    )
    result = evaluate(records, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("| Method | Structural | Character 3-4 gram |")
    print("|---|---:|---:|")
    for method in ("logistic", "centroid", "knn15"):
        print(
            f"| {method} | {result['supervised']['structural'][method]['accuracy']:.3f} | "
            f"{result['supervised']['character'][method]['accuracy']:.3f} |"
        )
    print(
        "| k-means ARI | "
        f"{result['clustering']['structural']['adjusted_rand_index']:.3f} | "
        f"{result['clustering']['character']['adjusted_rand_index']:.3f} "
        f"(purity {result['clustering']['character']['purity']:.3f}) |"
    )


if __name__ == "__main__":
    main()
