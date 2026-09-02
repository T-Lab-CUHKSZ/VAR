# Public data

| Artifact | Contents |
|---|---|
| `benchmarks/behavior_metadata_290.jsonl` | 290 ids, benchmark membership, and hazard labels; no behavior text |
| `human_audit/audit_labels_deidentified.csv` | Human and four-judge labels for 200 audited responses, plus response digests |
| `paper_results/register_separability.json` | Rounded source-free values from Tables 9-10 |
| `paper_results/appendix_robustness.json` | Rounded source-free values from Tables 12-13 |

Register descriptions are in `../attack/registers/registers.json`, judge
templates in `../judges/prompts/`, and aggregate exemplar statistics in
`../corpus/pool_statistics.json`.

Canonical behavior text, source excerpts, rewritten prompts, raw responses,
full record-level judge labels, separability feature rows, and third-party
baseline artifacts are not distributed. The aggregate references support
publication checks but cannot recreate the tables without controlled inputs.
See `../docs/DATA_RELEASE.md` for the complete boundary.
