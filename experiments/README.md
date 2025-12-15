# Experiments Layer

This directory hosts the reproducible experimentation stack for the NL->IR->SQL->DB->answer pipeline. The scope mirrors production behavior (same SQLite schema, same REST entrypoints) while isolating inputs/outputs so experiments never mutate the live configuration.

## Layout
- `config.yaml` - single source of truth for DB paths, model switches, endpoints, retry/temperature knobs, and artifact locations.
- `runner.py` - CLI orchestrator with subcommands `seed`, `run_baseline`, `run_system`, `evaluate`, `report`.
- `artifacts/` - run outputs (JSONL responses, metric JSON, generated reports). Created on demand.
- `README.md` - this file.

## Workflow
1. **Seed data** - `python experiments/runner.py seed`
   - By default `experiments/seed.py` generates multi-table demo data (people/projects/items/status_history plus alias dictionary) into `experiments/artifacts/experiments.db` and emits `experiments/artifacts/seed_summary.json`. Use `--seed/--items/--start-date/--end-date` for deterministic control; switch back to the legacy single-table seed via `--use-script` or `--script scripts/init_tasks_sqlite.py`.
2. **Run baseline** - `python experiments/runner.py run_baseline`
   - Stub by default; configure `baseline.command` in `config.yaml` to point at your classical baseline (for example, deterministic SQL template or zero-shot LLM prompt). The runner exports JSONL predictions to `artifacts/baseline_latest.jsonl` when the command is provided.
3. **Run system** - `python experiments/runner.py run_system`
   - Calls the existing `/tasks/ask` endpoint for each prompt in `system.question_file`, captures the raw JSON payloads, and writes them to the configured artifact. No KG/full-graph modules are toggled here.
4. **Evaluate** - `python experiments/runner.py evaluate`
   - Consumes the JSONL outputs, computes coverage metrics *and* optional exact-match stats when `data.reference_answers` (JSONL with `question`/`answer`) is provided. Override the path via `--reference`.
5. **Report** - `python experiments/runner.py report`
   - Reads the metrics file and produces a Markdown summary (for appendices/slides). Extend the template to auto-generate LaTeX tables, charts, etc.

All commands accept `--help` for overrides (question file, output path, and so on). See inline docstrings in `runner.py`.

## Adding new baselines/systems
- Keep the production APIs unchanged; a new experiment equals a new command in `config.yaml` pointing at a wrapper script.
- Never write to `app/` or `scripts/` from here unless you are extending logging/telemetry. If behavior diverges from prod, document it.
- Use JSONL for intermediate artifacts so we can diff runs and plug into notebooks easily.

## Automation hooks
- CI can call `python experiments/runner.py run_system --output ...` to capture regressions before deployment.
- When exporting figures, pipe `report` output into nbconvert/Typst as needed; keep raw data under `artifacts/` so reviewers can recompute metrics offline.
