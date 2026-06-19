"""EBrain VaultWatcher — watchdog-based file system monitor for Obsidian vault.

Fires a callback whenever a .md file in the vault changes.
Designed for detecting human edits (not programmatic VaultSync writes).

Caller pattern:
    watcher = VaultWatcher("/vault", on_change=my_callback)
    watcher.start()
    ...
    watcher.stop()

Requires: watchdog>=3.0 (install with `pip install ebrain[obsidian]`)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

_log = logging.getLogger("ebrain.vault_watcher")


class VaultWatcher:
    """Watch an Obsidian vault for .md file changes.

    Fires `on_change(path: Path)` in a background thread for every
    .md file modification. Caller is responsible for debounce/filtering.

    Requires `watchdog` package — raises ImportError with install hint if missing.
    """

    def __init__(
        self,
        vault_path: str | Path,
        on_change: Callable[[Path], None],
        *,
        patterns: list[str] | None = None,
    ) -> None:
        try:
            import watchdog.events  # noqa: F401
            import watchdog.observers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "watchdog is required for VaultWatcher. "
                "Install with: pip install 'ebrain[obsidian]'"
            ) from exc

        self.vault_path = Path(vault_path)
        self.on_change = on_change
        self.patterns = patterns or ["*.md"]
        self._observer: object | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start watching. Safe to call multiple times (idempotent)."""
        from watchdog.events import FileSystemEventHandler
        from watchdog.events import PatternMatchingEventHandler
        from watchdog.observers import Observer

        with self._lock:
            if self._observer is not None:
                return

            outer_self = self

            class _Handler(PatternMatchingEventHandler):
                def __init__(self) -> None:
                    super().__init__(
                        patterns=outer_self.patterns,
                        ignore_directories=True,
                        case_sensitive=False,
                    )

                def on_modified(self, event: FileSystemEventHandler) -> None:
                    path = Path(str(event.src_path))
                    _log.debug("vault change: %s", path)
                    try:
                        outer_self.on_change(path)
                    except Exception:
                        _log.exception("on_change callback raised for %s", path)

                def on_created(self, event: FileSystemEventHandler) -> None:
                    self.on_modified(event)

            observer = Observer()
            observer.schedule(_Handler(), str(self.vault_path), recursive=True)
            observer.start()
            self._observer = observer
            _log.info("VaultWatcher started: %s", self.vault_path)

    def stop(self) -> None:
        """Stop watching and release resources."""
        with self._lock:
            if self._observer is None:
                return
            obs = self._observer
            self._observer = None

        obs.stop()  # type: ignore[attr-defined]
        obs.join()  # type: ignore[attr-defined]
        _log.info("VaultWatcher stopped")

    def __enter__(self) -> "VaultWatcher":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
