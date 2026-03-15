# ref-engine

REF -- Recognitive Equation Framework for Agentic Validation.

## Overview

`ref-engine` is a Python-based analysis and reporting system for ingesting files, extracting features, running scoring and analytics, and exporting structured reports.

## Proposed architecture

- `src/ref_engine/ref.py` -- orchestration / entrypoint
- `src/ref_engine/file_parser.py` -- file ingestion
- `src/ref_engine/corpus_injector.py` -- corpus normalization / loading
- `src/ref_engine/feature_extractor.py` -- feature extraction
- `src/ref_engine/analytics.py` -- analytics layer
- `src/ref_engine/scoring_engine.py` -- scoring logic
- `src/ref_engine/quaternion.py` -- math utilities
- `src/ref_engine/json_exporter.py` -- structured JSON export
- `src/ref_engine/analytics_renderer.py` -- analytics rendering
- `src/ref_engine/report_renderer.py` -- report rendering
- `src/ref_engine/pipeline.py` -- agentic pipeline runner
- `src/ref_engine/differential_analyser.py` -- corpus differential analysis

## Install

```bash
make install
```

## Test

```bash
make test
```

## Lint

```bash
make lint
```

## Repository standards

This repo includes:
- CI via GitHub Actions
- project guidance via CLAUDE.md
- contribution and security docs
- issue and pull request templates
- agent-native documentation structure

## Status

Initial bootstrap in progress.
