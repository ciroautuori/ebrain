# SESSION STATE — ebrain
> auto-maintained by auto-optimizer. Loaded each SessionStart so the thread is never lost.
> Edit freely — the loop refreshes the footer + handoff index, your tasks/facts persist.

## Open tasks (carried across sessions)
- [ ] Add GitHub Actions CI (test + ruff + mypy) — file `.github/workflows/ci.yml`
- [ ] Real migration system (Alembic or yoyo) — replace `ensure_schema()` CREATE IF NOT EXISTS
- [ ] Add `py.typed` marker (PEP 561) — touch `src/ebrain/py.typed` + add to pyproject
- [ ] Add `__aenter__`/`__aexit__` on MemoryPipeline for clean pool shutdown

## Handoff index (newest first)
- (none yet)
## Durable project facts
- Qdrant recall is completely broken (v0.1.0) — always falls back to keyword search
- dream.py has wrong table names — crashes on first use
- All env vars for memory config use EROS_MEMORY_* prefix (leftover from EROS project), not EBRAIN_*
- 22 tests exist but all unit-only, zero integration coverage

## Next proposal (discuss)
- Obsidian integration: vault sync layer — EBrain↔Obsidian bidirectional, mobile via Syncthing/iCloud
- New module: `src/ebrain/vault.py` — VaultSync class, converts memories→.md + watches for edits
- Folder structure: ebrain/memories/, scenes/, personas/, graph/entities/

_last refresh: 2026-06-19T10:43:17Z · branch main · 7a19a79_
