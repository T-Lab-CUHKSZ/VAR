# De-identified blind-audit data

`audit_labels_deidentified.csv` contains labels for the 200-response
blind audit reported in the paper. It includes no request text, response text,
prompt identifier, server path, or annotator note. The response SHA-256 column
lets an authorized holder of the controlled texts verify an exact match without
redistributing those texts.

The 2-of-4 ensemble confusion matrix against the human reference is
TP=99, FP=1, FN=48, TN=52. This gives precision 0.990, recall 0.673, raw
agreement 0.755, Cohen's kappa 0.510, PABAK 0.510, and Gwet's AC1 0.536.

See `human_eval/INSTRUCTIONS.md` for the annotation protocol and
`human_eval/score_blind_audit.py` for validation and metric computation.
