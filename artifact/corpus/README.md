# AO3 corpus construction

The experiment used 355 manually screened public excerpts across twelve
registers. Each excerpt is at most 350 characters, and five are sampled
uniformly for each rewriter call. Excerpt text is not redistributed.

The controlled workflow is:

1. `collect_ao3.py` collects public candidate works without login. It checks
   `robots.txt`, requires a contact-bearing user agent, honors backoff, and
   defaults to an eight-second delay.
2. `prepare_exemplar_review.py` creates a private sheet with one candidate
   excerpt per work.
3. A reviewer completes `include`, `pii_found`,
   `identifying_metadata_found`, and `quality_ok`.
4. `build_exemplar_pool.py` accepts reviewed, PII-free rows and enforces the
   reported register counts and character cap.

Before collection, confirm that applicable site terms and access rules permit the
intended use. Keep raw pages, review sheets, and the exemplar JSONL outside the
repository. Aggregate statistics are in `pool_statistics.json`.

```bash
python -m pip install -r requirements-data.txt
python corpus/collect_ao3.py \
  --raw-dir controlled/ao3_raw \
  --user-agent "VAR research client; contact researcher@example.org"
python corpus/prepare_exemplar_review.py \
  --raw-dir controlled/ao3_raw \
  --out controlled/ao3_review.csv
python corpus/build_exemplar_pool.py \
  --review controlled/ao3_review.csv \
  --out controlled/exemplars_12reg.jsonl \
  --stats-out controlled/exemplar_statistics.json
```
