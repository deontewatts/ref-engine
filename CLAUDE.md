# ref-engine Agent Guide

Use this file as the default onboarding context for this repository.

## Why

This repository implements REF -- Recognitive Equation Framework for Agentic Validation. The project ingests files, extracts structured meaning, computes analytics and scores, and produces reports and exports.

## What

- `src/ref_engine/` -- application code
- `tests/` -- tests
- `docs/agent-guides/` -- deeper operational guidance
- `.github/` -- CI, issue templates, PR templates
- `.claude/skills/` -- task-specific agent playbooks

## How

- Use existing patterns before introducing new abstractions.
- Validate changes with lint, test, and build-oriented checks.
- Keep generated artifacts out of source unless intentionally versioned.
- Prefer modular, typed, maintainable Python code.

## Progressive Disclosure

Read only what is relevant:
- build/test/lint: `docs/agent-guides/build-test-verify.md`
- conventions: `docs/agent-guides/core-conventions.md`
- architecture: `docs/architecture.md`

Use skills in `.claude/skills/` when relevant.
