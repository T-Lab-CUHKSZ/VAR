# Voices Across Registers

Reference implementation for:

> **Voices Across Registers: Corpus-Conditioned Vernacular Jailbreaks against
> Aligned LLMs via Fanfiction Subgenres**

The package covers corpus construction, attack construction, target inference,
four-judge evaluation, blind human validation, defenses, and analysis. It is a
method-audit reference, not a turnkey attack package. Source-derived excerpts,
canonical behavior text, rewritten prompts, and raw target outputs are withheld
under the responsible-release policy in `docs/DATA_RELEASE.md`.

## Method summary

- **Vernacular experiment:** twelve AO3-derived registers, six existing
  single-turn templates, and six length-matched PlanX controls over 290 unique
  behaviors and eight target models.
- **Factorial experiment:** seven structural conditions crossed with twelve
  registers, yielding 84 cells.
- **VAR-A4:** T1 is the stored register-conditioned entry for the selected
  behavior-register record. Fixed T2-T4 turns expand the scene, increase
  specificity, and compile the preceding material. The four replies are
  concatenated and judged as one output.
- **Evaluation:** HarmBench-13B, LlamaGuard-3-8B, WildGuard-7B, and
  GPT-5.4-mini/StrongREJECT. Attack success is the two-of-four ensemble label.

VAR-A4 uses no attacker LLM or optimization during the primary interaction.
The adaptive continuation is a separate ablation. FITD, ActorAttack, and
Crescendo are represented by the static four-turn adapters used in the paper;
third-party repositories and datasets are not included.

## Contents

| Directory | Scope |
|---|---|
| `attack/` | Five-shot meta-prompt, factorial builder, PlanX, VAR-A4, and Table 7 variants |
| `corpus/` | Rate-limited AO3 collection, manual-screening preparation, and pool validation |
| `inference/` | Six open-weight targets and two provider-API targets |
| `judges/` | Three local judges, StrongREJECT, and Appendix L.1 templates |
| `defenses/` | Self-Reminder and SmoothLLM |
| `baselines/` | Static FITD, ActorAttack, and Crescendo adapters |
| `human_eval/` | Blinded sampling, annotation protocol, and scoring |
| `analysis/` | Tables 1-7, 9-10, 12-13, 16, Figure 5, and human-audit analysis |
| `data/` | Public metadata, de-identified audit labels, and aggregate references |
| `scripts/` | Publication-boundary validator |
| `tests/` | Method-contract tests |

Key paper mappings:

| Paper component | Implementation |
|---|---|
| Sections 3.1-3.3 | `attack/meta_prompt.py`, `attack/factorial_meta.py` |
| Section 3.4 and Appendix L.2 | `attack/var_a4.py` |
| Section 3.5 and Appendix L.1 | `judges/` |
| Section 4 | `inference/` |
| Tables 1-7 and 16 | Corresponding scripts in `analysis/` |
| Tables 9-10 and 12-13 | Separability and robustness scripts in `analysis/` |
| External multi-turn comparison | `baselines/` |
| Blind human validation | `human_eval/`, `analysis/human_audit.py` |
| Table 8 corpus statistics | `corpus/pool_statistics.json` |

## Environment

Use Python 3.10 or newer. Core analysis and the Linux/CUDA runtime are kept in
separate requirement files. vLLM resolves its compatible Transformers
dependency and supports the open-weight target and local-judge runners.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-runtime.txt  # Linux/CUDA only
```

Open-weight checkpoints default to canonical Hugging Face identifiers. Set
`MODEL_ROOT` to use local checkpoints:

```bash
export MODEL_ROOT=/path/to/checkpoints
```

Provider-API runs read credentials and optional gateway URLs from the
environment:

```bash
export GEMINI_API_KEY=...
export GEMINI_BASE_URL=...
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=...
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

Local judge runners default to public checkpoint identifiers and accept
checkpoint overrides through command-line flags.

## Record contract

Each inference input requires `prompt_id`, non-empty `turns`, and canonical
`behavior` text. Inference writes `responses` and `combined_response`; judges
score `combined_response`. For VAR-A4 this is the concatenated T1-T4 target
output. Empty outputs receive a failed-attack label.

Open-weight generation defaults to 2,048 output tokens, temperature 0.7, and
top-p 0.9, with a stable record-level seed derived from `--seed`. Provider-API
targets use provider-default sampling.

## Data and verification

The public data include metadata for 290 behaviors, aggregate statistics for
the 355-excerpt pool, a de-identified 200-row audit table, judge templates,
and source-free reference values for Tables 9-10 and 12-13. Controlled inputs
are required to rerun the full numerical pipeline. See `docs/REPRODUCE.md` for
commands and `docs/DATA_RELEASE.md` for the complete release boundary.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python scripts/validate_release.py
```

The validator checks reported dimensions and schemas, method contracts,
aggregate references, and the absence of private paths or unexpected raw data.

## License

MIT. Referenced models and benchmark artifacts remain governed by their
original licenses.
