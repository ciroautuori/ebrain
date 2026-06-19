"""EBrain Vault Daemon — watches raw/ subfolder, auto-ingests new files.

Karpathy pattern: you drop files in {vault}/raw/, the daemon picks them up,
ingests them via LLM, writes wiki pages + entity graph, updates index.md.

Usage:
    from ebrain.vault_daemon import run_daemon
    await run_daemon("/path/to/ObsidianVault")

CLI:
    ebrain vault watch --vault /path/to/vault   (via cli.py)

Tracks ingested files in {vault}/ebrain/.ingested to avoid re-processing.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

_log = logging.getLogger("ebrain.vault_daemon")

# File extensions to ingest
INGEST_EXTENSIONS = {".md", ".txt"}

# Folder inside vault that contains source documents
RAW_SUBDIR = "raw"


class VaultDaemon:
    """Watch vault/raw/ for new files and auto-ingest them into EBrain."""

    def __init__(self, vault_path: str | Path) -> None:
        self.vault_path = Path(vault_path)
        self.raw_dir = self.vault_path / RAW_SUBDIR
        self._ingested_file = self.vault_path / "ebrain" / ".ingested"
        self._ingested: set[str] = set()
        self._stop = asyncio.Event()

    def _load_ingested(self) -> None:
        if self._ingested_file.exists():
            self._ingested = set(
                line.strip()
                for line in self._ingested_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

    def _mark_ingested(self, path: Path) -> None:
        key = str(path.relative_to(self.vault_path))
        self._ingested.add(key)
        self._ingested_file.parent.mkdir(parents=True, exist_ok=True)
        with self._ingested_file.open("a", encoding="utf-8") as fh:
            fh.write(key + "\n")

    def _is_ingested(self, path: Path) -> bool:
        key = str(path.relative_to(self.vault_path))
        return key in self._ingested

    async def _ingest_file(self, path: Path, vault: object) -> None:
        from ebrain.vault_ingest import ingest_source
        _log.info("ingesting: %s", path.name)
        try:
            result = await ingest_source(path, vault)  # type: ignore[arg-type]
            _log.info(
                "done: %s — %d entities, %d facts",
                path.name, result.get("entities_added", 0), result.get("key_facts_count", 0),
            )
            print(
                f"  ✓ {path.name:<50} "
                f"+{result.get('entities_added', 0)}ent "
                f"+{result.get('key_facts_count', 0)}facts"
            )
            self._mark_ingested(path)
        except Exception as exc:
            _log.error("ingest failed: %s — %s", path.name, exc)
            print(f"  ✗ {path.name}: {exc}")

    async def scan_and_ingest(self) -> int:
        """Scan raw/ for new files, ingest them. Returns count ingested."""
        if not self.raw_dir.exists():
            return 0

        from ebrain.vault import VaultSync
        vault = VaultSync(self.vault_path)

        pending = [
            f for f in sorted(self.raw_dir.rglob("*"))
            if f.is_file()
            and f.suffix.lower() in INGEST_EXTENSIONS
            and not self._is_ingested(f)
            and f.stat().st_size > 50
        ]

        if not pending:
            return 0

        _log.info("found %d new files to ingest", len(pending))
        for path in pending:
            await self._ingest_file(path, vault)

        if pending:
            vault.update_index()

        return len(pending)

    async def run(self, poll_seconds: float = 5.0) -> None:
        """Poll raw/ every poll_seconds for new files. Runs until stop()."""
        self._load_ingested()
        print(f"\nEBrain Vault Daemon — watching {self.raw_dir}")
        print(f"Drop files in {self.raw_dir}/ to auto-ingest.")
        print("Ctrl+C to stop.\n")

        # Initial scan
        n = await self.scan_and_ingest()
        if n > 0:
            print(f"  Initial scan: {n} files ingested.\n")

        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
            if not self._stop.is_set():
                await self.scan_and_ingest()

        print("Daemon stopped.")

    def stop(self) -> None:
        self._stop.set()


async def run_daemon(vault_path: str | Path, poll_seconds: float = 5.0) -> None:
    """Start the vault daemon. Blocks until Ctrl+C."""
    import signal
    daemon = VaultDaemon(vault_path)
    loop = asyncio.get_event_loop()

    def _shutdown(*_: object) -> None:
        daemon.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    await daemon.run(poll_seconds=poll_seconds)
