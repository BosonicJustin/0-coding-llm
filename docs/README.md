# Documentation

This directory contains the authoritative design, operating, and experiment
records for the project. Start with the operational handoff for live state and
the production runbook for ordered execution.

## Operations

- [Current handoff](operations/handoff.md)
- [Pre-training readiness checklist](operations/pretraining-checklist.md)
- [Production runbook](operations/production-runbook.md)
- [RunPod six-GPU pod qualification](operations/runpod-six-gpu-pod-qualification.md)
- [Six-GPU geometry evidence producer](operations/six-gpu-geometry-evidence-producer.md)
- [Final training-corpus qualification](operations/training-corpus-qualification.md)

## Data pipeline

- [End-to-end data pipeline](data/data-pipeline.md)
- [Current fast corpus generation v2](data/fast-generation-v2.md)
- [Fast all-eligible curation runbook](data/fast-all-eligible-curation.md)
- [Stack and English curation contract](data/curation.md)
- [Local-WAL curation acceleration](data/curation-acceleration.md)
- [Streaming raw audit](data/streaming-preprocess.md)
- [Curation-independent raw token cache](data/raw-token-cache.md)
- [Raw-token-cache materializer integration](data/raw-token-cache-materializer-integration.md)
- [Selection-v7 packed-supply gate](data/selection-v7-supply-gate.md)
- [FineWeb-Edu acquisition](data/english-pipeline.md)
- [Wikipedia acquisition](data/wikipedia-pipeline.md)
- [Optional English near-deduplication](data/english-near-dedup.md)
- [English near-deduplication calibration](data/english-near-dedup-calibration.md)
- [Curation-to-packed materialization](data/materialization.md)

## Pre-training

- [Model architecture](training/model.md)
- [Packed training data and loader](training/training-data.md)
- [Training harness](training/training.md)
- [Immutable pre-training run authority](training/pretraining-run-authority.md)
- [One-chunk overfit and exact-resume qualification](operations/one-chunk-overfit-qualification.md)
- [Six-GPU launch qualification](operations/six-gpu-launch-qualification.md)

## Post-training

- [Post-training dataset acquisition and quarantine](posttraining/posttraining-data.md)
- [Prime Intellect SFT integration](posttraining/prime-sft.md)

## Experiment record

- [Experiment log](experiment/experiment-log.md)

Component-specific documentation remains next to the component:

- [Prime integration assets](../integrations/prime_intellect/README.md)
- [Coding smoke environment](../environments/coding_smoke/README.md)
