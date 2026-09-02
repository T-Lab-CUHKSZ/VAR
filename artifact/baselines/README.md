# Multi-turn baseline adapters

`adapters.py` defines the static four-turn FITD, ActorAttack, and Crescendo
variants used in Tables 6 and 16. `build_multiturn.py` writes records accepted
by the shared inference and judge pipeline.

```bash
python baselines/build_multiturn.py \
  --behaviors /controlled/behaviors.jsonl \
  --method fitd --method actorattack --method crescendo \
  --out /controlled/multiturn_baselines.jsonl
```

Use `normalize_external.py` to map separately authorized third-party records
to the same schema. These adapters do not reproduce an external adaptive
controller, and third-party code, data, licenses, and usage restrictions remain
separate.

Output fields include `prompt_id`, canonical `behavior`, behavior metadata,
`baseline_family`, `adapter_variant`, `turn_count`, and `turns`.
