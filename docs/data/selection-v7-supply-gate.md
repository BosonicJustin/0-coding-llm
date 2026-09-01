# Selection-v7 packed-supply gate

Run this cheap, manifest-only gate immediately after publishing selection v7
and before cache-backed materialization. It prevents reading and writing the
full corpus when one split/domain cannot fill the intended 40% Python, 40%
other-code, 20% English packed-input cap.

The gate authenticates the exact `manifest.json` bytes with the canonical
`manifest.sha256` sidecar, requires the production all-eligible v7 contract,
and invokes the materializer's strict selected-total validator. It does not
trust a loose `jq` projection: the selected totals, reference quotas, and
document totals must reconcile.

For each split/domain, the no-padding packer has

```text
stream_tokens = selected_content_tokens + documents
available_rows = floor((stream_tokens - 1) / sequence_length)
```

There is one EOS boundary token per complete document. The subtraction is
required because each causal row needs a following lookahead token; adjacent
rows share that lookahead. The gate then uses the training-order builder's
same stable largest-remainder allocator at 40/40/20. This is deliberately
geometry-independent and conservative for train: later optimizer-update
rounding can only reduce the required row count.

Use the CPU qualification environment because the command imports the exact
materializer/order validators, whose production module graph includes
PyTorch:

```bash
PROJECT_ROOT=/workspace/0-coding-llm
DATA_V2=/workspace/dataset-other-code-topup-v2
QUAL_PY=/opt/coding-model-qualification-venv/bin/python
SELECTION_ROOT="$DATA_V2/curated/selection-v7"

"$QUAL_PY" -u "$PROJECT_ROOT/scripts/qualify_selection_supply.py" \
  --selection-root "$SELECTION_ROOT" \
  --sequence-length 4096 \
  --expected-train-input-tokens 52580000000 \
  --expected-validation-input-tokens 500000000 \
  --expected-test-input-tokens 500000000
```

Exit status `0` means every split/domain has enough complete packed rows.
Status `1` means the authenticated selection is valid but has a supply
shortfall; stdout contains exact missing rows and input tokens. Status `2`
means the selection authority or arguments are invalid. Stdout is always one
machine-readable JSON object.

At sequence length 4,096, the geometry-independent requirements are:

| Split | Python rows | Other-code rows | English rows | Total rows | Maximum input tokens |
|---|---:|---:|---:|---:|---:|
| train | 5,134,766 | 5,134,765 | 2,567,383 | 12,836,914 | 52,579,999,744 |
| validation | 48,828 | 48,828 | 24,414 | 122,070 | 499,998,720 |
| test | 48,828 | 48,828 | 24,414 | 122,070 | 499,998,720 |

The 256-token train difference and 1,280-token held-out differences are the
unavoidable result of selecting only complete 4,096-token rows. Final train
order construction may be slightly smaller again after the measured six-GPU
optimizer-update geometry is frozen.

This preflight is not a final corpus qualification receipt. Materialization
still re-authenticates the decision bitmaps and raw/cache inputs, reconciles
the actual packed manifests to these selected totals, and the final corpus
qualifier must pass after the orders are published.
