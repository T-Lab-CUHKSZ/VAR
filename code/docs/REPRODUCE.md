# Reproduction and verification

These commands assume access to separately authorized controlled inputs. Run
the release validator before publishing or archiving the package:

```bash
python scripts/validate_release.py
```

## Record contract

Each prompt JSONL row contains:

```text
prompt_id       unique record identifier
behavior        canonical behavior text for judge conditioning
behavior_id     public id, b000 through b289
behavior_idx    integer index, 0 through 289
benchmark       HarmBench or JailbreakBench
hazard          one of sixteen reported categories
arm             experimental arm
turns           list of user turns
turn_count      length of turns
```

`behavior_id` cannot replace the canonical `behavior` text used by HarmBench
and StrongREJECT.

## 1. Prompt construction

The controlled corpus contains 355 screened excerpts across twelve registers,
each capped at 350 characters. See `corpus/README.md` for the collection,
manual review, and pool-validation workflow.

```bash
python attack/factorial_meta.py \
  --exemplars /controlled/exemplars_12reg.jsonl \
  --behaviors /controlled/factorial_30.jsonl \
  --out /controlled/factorial_7x12.jsonl
```

`attack/plan_x.py` implements the Appendix L.3 length-matched rewriter.

## 2. VAR-A4 and Table 7 variants

VAR-A4 uses the stored register-conditioned T1 for each behavior-register
record and the fixed T2-T4 sequence:

```bash
python attack/var_a4.py \
  --register_entries /controlled/register_entries.jsonl \
  --out /controlled/var_a4_290.jsonl
python attack/var_a4_ablations_factorial.py \
  --var_prompts /controlled/var_a4_290.jsonl \
  --out_dir /controlled/var_a4_factorial
python attack/var_a4_ablations_dup.py \
  --var_prompts /controlled/var_a4_290.jsonl \
  --out_dir /controlled/var_a4_duplication
```

The structural builder emits all eight T2/T3/T4 cells. The duplication builder
emits the four five-turn variants in the lower block of Table 7.

## 3. Multi-turn baselines

```bash
python baselines/build_multiturn.py \
  --behaviors /controlled/behaviors.jsonl \
  --method fitd --method actorattack --method crescendo \
  --out /controlled/multiturn_baselines.jsonl
```

These are the static four-turn variants used in Tables 6 and 16. Use
`baselines/normalize_external.py` to map separately authorized records to the
common `prompt_id`/`behavior`/`turns` schema.

## 4. Target inference

Open-weight targets use temperature 0.7, top-p 0.9, a 2,048-token output cap,
and a stable seed derived per record:

```bash
python inference/eval_open_weight.py \
  --prompts /controlled/prompts.jsonl \
  --model qwen3.5-9b \
  --out /controlled/responses \
  --seed 42
```

Provider-API targets use provider-default sampling:

```bash
python inference/eval_frontier.py \
  --prompts /controlled/prompts.jsonl \
  --model gemini-2.5-pro \
  --out /controlled/responses/gemini-2.5-pro_responses.jsonl
```

Pass `--behavior_filter` with a JSON list of `behavior_idx` values for a
stratified provider-API subset.

## 5. Four-judge evaluation

Judges score `combined_response`. For VAR-A4 this is the concatenated T1-T4
target output; for a single-turn record it is the sole response.

```bash
python judges/run_local_judges.py \
  --responses /controlled/responses/qwen3.5-9b_responses.jsonl \
  --out /controlled/judges/qwen3.5-9b
python judges/run_strongreject_judge.py \
  --responses /controlled/responses/qwen3.5-9b_responses.jsonl \
  --out /controlled/judges/qwen3.5-9b/strongreject.jsonl
```

Empty outputs receive failed-attack labels. Parser or infrastructure failures
remain incomplete and must be retried. Attack success is the two-of-four
ensemble decision.

## 6. Analysis

| Script | Paper output |
|---|---|
| `analysis/vernacular_asr.py` | Table 1 |
| `analysis/factorial_amplification.py` | Table 2 |
| `analysis/factorial_gee.py` | Figure 5 |
| `analysis/length_gee.py` | Table 3 |
| `analysis/judge_agreement.py` | Table 4 and Appendix C |
| `analysis/defense_ratio.py` | Table 5 |
| `analysis/multiturn_comparison.py` | Table 6 |
| `analysis/var_factorial.py`, `analysis/var_dup_sensitivity.py` | Table 7 |
| `analysis/register_separability.py` | Tables 9-10 |
| `analysis/bootstrap_ci.py` | Table 12 |
| `analysis/threshold_robustness.py` | Table 13 |
| `analysis/per_hazard.py` | Table 16 |
| `analysis/human_audit.py` | Blind-validation table |

Headline summaries macro-average across eight targets. Table 16
micro-averages all available model-behavior observations within each benchmark
and hazard.

### Robustness

The behavior-cluster bootstrap uses 10,000 repetitions, root seed 42, and one
child stream per target. The threshold sweep evaluates StrongREJECT cutoffs
0.125, 0.20, 0.25, 0.30, and 0.375.

```bash
python analysis/bootstrap_ci.py \
  --prompts /controlled/register_prompts.jsonl \
  --judge-root /controlled/judges \
  --out /controlled/bootstrap_table12.json
python analysis/threshold_robustness.py \
  --prompts /controlled/register_prompts.jsonl \
  --judge-root /controlled/judges \
  --out /controlled/threshold_table13.json
```

Both scripts require four judge records by default. The optional `invariant`
policy accepts a three-vote row only when either value of the missing vote
leaves the ensemble decision unchanged.

### Register separability

Tables 9-10 use 290 behaviors by 12 registers. The behavior-level split is
203/87, or 2,436/1,044 prompts. Classifiers use either fifteen structural
features or 6,000 L2-normalized character 3-4-gram features; twelve-cluster
k-means is evaluated with ARI and purity.

Perplexity is the fifteenth structural feature. The caller must select an
authorized causal-language-model checkpoint:

```bash
python analysis/compute_perplexity.py \
  --prompts /controlled/register_prompts.jsonl \
  --model /authorized/perplexity-checkpoint \
  --out /controlled/prompt_perplexity.jsonl
python analysis/register_separability.py \
  --prompts /controlled/register_prompts.jsonl \
  --perplexity /controlled/prompt_perplexity.jsonl \
  --out /controlled/register_separability.json
```

Source-free rounded values for Tables 9-10 and 12-13 are in
`data/paper_results/`; they are validation references, not record-level inputs.

## 7. Blind human validation

`prepare_blind_audit.py` joins eight response files with four complete judge
labels, samples 100 ensemble-success and 100 ensemble-non-success records, and
checks coverage of all registers, targets, and hazards. Keep its private key
hidden from the annotator.

```bash
python human_eval/prepare_blind_audit.py \
  --response qwen3.5-9b=/controlled/responses/qwen3.5-9b_responses.jsonl \
  --response qwen3.5-4b=/controlled/responses/qwen3.5-4b_responses.jsonl \
  --response llama-3.1-8b=/controlled/responses/llama-3.1-8b_responses.jsonl \
  --response llama-3.2-3b=/controlled/responses/llama-3.2-3b_responses.jsonl \
  --response gemma-4-E2B=/controlled/responses/gemma-4-E2B_responses.jsonl \
  --response gemma-4-E4B=/controlled/responses/gemma-4-E4B_responses.jsonl \
  --response gemini-2.5-pro=/controlled/responses/gemini-2.5-pro_responses.jsonl \
  --response deepseek-v4-pro=/controlled/responses/deepseek-v4-pro_responses.jsonl \
  --judge-root /controlled/judges \
  --out-blind /controlled/blind_audit.csv \
  --out-key /controlled/audit_key.jsonl
```

After annotation, validate labels, compute agreement, and create the public
table:

```bash
python human_eval/score_blind_audit.py \
  --annotations /controlled/blind_audit_completed.csv \
  --key /controlled/audit_key.jsonl \
  --out-summary /controlled/audit_summary.json \
  --out-release data/human_audit/audit_labels_deidentified.csv
```

Annotation definitions are in `human_eval/INSTRUCTIONS.md`.
