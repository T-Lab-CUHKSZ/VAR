# Data release policy

## Released artifacts

| Artifact | Scope |
|---|---|
| Source code | Corpus and attack construction, inference, four judges, blind validation, defenses, and analysis |
| Register and behavior metadata | Twelve register descriptions; 290 behavior ids, benchmark membership, and hazard labels |
| Exemplar statistics | Source-free counts and word ranges for the 355-row pool |
| Blind-audit data | Human and judge labels plus response digests for 200 samples; no text or prompt identifiers |
| Prompt and method definitions | Judge templates, VAR-A4 construction, PlanX, and static multi-turn adapters |
| Appendix analysis | Bootstrap, threshold robustness, and register-separability code |
| Aggregate references | Rounded source-free values from Tables 9-10 and 12-13 |

## Withheld artifacts

| Artifact | Reason |
|---|---|
| Source-derived AO3 excerpts | Source text is not redistributed |
| Canonical behavior text | Avoids publishing a turnkey prompt-generation package |
| Rewritten prompts and raw responses | May be copyable or operationally sensitive |
| Full record-level judge labels | May identify vulnerable behavior-register-model combinations |
| Separability prompt and feature rows | Contain controlled prompts or record-level measurements |
| Third-party baseline code and data | Remain governed by their original licenses and terms |

Worked examples mask operational spans. The public blind-audit table similarly
contains labels and response digests but no request or response text. Request
digests are omitted because the controlled request set is small enough for
dictionary matching.

## Verification scope

The package exposes transformations, generation settings, judge rules,
statistical estimators, and experimental dimensions. It supports method
inspection and reruns with separately authorized inputs. The validator pins
key schemas, method contracts, and aggregate appendix values.

The package alone cannot reproduce all numerical tables; that requires the
controlled exemplar pool, behavior text, target outputs, and judge labels.

## Responsible use

Keep excerpts, rewritten prompts, and raw responses access-controlled; redact
operational spans in examples; evaluate only authorized systems; and follow
the paper's Ethics Statement and applicable disclosure practices.
