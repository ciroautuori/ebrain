#!/usr/bin/env python3
"""EBrain Setup Wizard — installs and configures EBrain from scratch.

What this does:
  1. Check prerequisites (Docker, Python 3.12+, opencode)
  2. Start PostgreSQL + Qdrant containers (or use existing)
  3. Create ebrain DB user + database
  4. Run EBrain migrations
  5. Configure EBRAIN_* env vars (writes to ~/.config/ebrain/env)
  6. Test ask_json via opencode/deepseek-v4-flash-free (FREE)
  7. Optional: configure Obsidian vault path
  8. Run a quick smoke test

Usage:
    python3 scripts/setup.py
    python3 scripts/setup.py --vault /path/to/ObsidianVault
    python3 scripts/setup.py --non-interactive  # use all defaults
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Colors ────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}✓{RESET}  {msg}")


def err(msg: str) -> None:
    print(f"{RED}✗{RESET}  {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"{YELLOW}!{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"{CYAN}→{RESET}  {msg}")


def header(title: str) -> None:
    print(f"\n{BOLD}{'─' * 55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 55}{RESET}")


# ── Prerequisites ─────────────────────────────────────────────────────────────

def check_python() -> bool:
    v = sys.version_info
    if v < (3, 12):
        err(f"Python 3.12+ required (found {v.major}.{v.minor})")
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_docker() -> bool:
    if shutil.which("docker") is None:
        err("Docker not found. Install from https://docs.docker.com/get-docker/")
        return False
    r = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    if r.returncode != 0:
        err("Docker daemon not running. Start Docker first.")
        return False
    ok("Docker running")
    return True


def check_opencode() -> tuple[bool, str]:
    """Return (found, model). Tries opencode, falls back to claude -p."""
    if shutil.which("opencode"):
        ok("opencode found")
        return True, "opencode/deepseek-v4-flash-free"
    warn("opencode not found")
    info("Install: npm install -g opencode-ai  (or see https://opencode.ai)")
    if shutil.which("claude"):
        warn("Falling back to claude -p (Claude Max required)")
        return True, "claude"
    err("No LLM backend found. Install opencode or claude.")
    return False, ""


def install_ebrain() -> bool:
    """Install ebrain package in editable mode if not already installed."""
    try:
        import ebrain  # noqa: F401
        ok("ebrain package importable")
        return True
    except ImportError:
        pass
    info("Installing ebrain...")
    project_root = Path(__file__).parent.parent
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(project_root), "--quiet"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        err(f"pip install failed: {r.stderr[:200]}")
        return False
    ok("ebrain installed")
    return True


# ── Docker containers ─────────────────────────────────────────────────────────

def container_running(name: str) -> bool:
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True, text=True
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def start_postgres(
    container_name: str = "ebrain-db",
    port: int = 5432,
    password: str = "ebrain",
) -> bool:
    if container_running(container_name):
        ok(f"PostgreSQL container '{container_name}' already running")
        return True
    info(f"Starting PostgreSQL container '{container_name}' on port {port}...")
    r = subprocess.run([
        "docker", "run", "-d",
        "--name", container_name,
        "--restart", "unless-stopped",
        "-e", f"POSTGRES_PASSWORD={password}",
        "-e", "POSTGRES_USER=ebrain",
        "-e", "POSTGRES_DB=ebrain",
        "-p", f"{port}:5432",
        "pgvector/pgvector:pg17",
    ], capture_output=True, text=True)
    if r.returncode != 0:
        # Container might already exist but stopped
        subprocess.run(["docker", "start", container_name], capture_output=True)
    import time
    time.sleep(3)
    if container_running(container_name):
        ok(f"PostgreSQL running on port {port}")
        return True
    err(f"Failed to start PostgreSQL: {r.stderr[:200]}")
    return False


def start_qdrant(
    container_name: str = "ebrain-qdrant",
    port: int = 6333,
) -> bool:
    if container_running(container_name):
        ok(f"Qdrant container '{container_name}' already running")
        return True
    info(f"Starting Qdrant container '{container_name}' on port {port}...")
    r = subprocess.run([
        "docker", "run", "-d",
        "--name", container_name,
        "--restart", "unless-stopped",
        "-p", f"{port}:6333",
        "qdrant/qdrant:latest",
    ], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["docker", "start", container_name], capture_output=True)
    import time
    time.sleep(2)
    if container_running(container_name):
        ok(f"Qdrant running on port {port}")
        return True
    err(f"Failed to start Qdrant: {r.stderr[:200]}")
    return False


# ── LLM backend ───────────────────────────────────────────────────────────────

async def _ask_json_opencode(prompt: str, model: str) -> dict:
    """Call opencode run -m <model> with prompt, extract JSON from output."""
    result = await asyncio.to_thread(
        subprocess.run,
        ["opencode", "run", "-m", model, "--pure",
         prompt + "\n\nReturn ONLY valid JSON. No markdown fences. Start with { end with }."],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout

    # Extract JSON object (opencode may output extra text)
    start = output.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(output[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(output[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def configure_llm(model: str) -> None:
    """Wire ask_json to opencode/deepseek-v4-flash-free."""
    from ebrain.llm import set_ask_json

    async def _ask(prompt: str) -> dict:
        return await _ask_json_opencode(prompt, model)

    set_ask_json(_ask)
    ok(f"ask_json configured → {model}")


# ── Env config ────────────────────────────────────────────────────────────────

ENV_FILE = Path.home() / ".config" / "ebrain" / "env"


def write_env(config: dict) -> None:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# EBrain config — generated by setup.py\n"]
    for k, v in config.items():
        lines.append(f'export {k}="{v}"\n')
    ENV_FILE.write_text("".join(lines))
    ok(f"Env written to {ENV_FILE}")
    info(f"Add to ~/.bashrc or ~/.zshrc:  source {ENV_FILE}")


# ── Schema & smoke test ───────────────────────────────────────────────────────

async def run_migrations(database_url: str) -> bool:
    import ebrain.db as db
    db.DATABASE_URL = database_url
    db._pool = None
    try:
        from ebrain.migrations import run_migrations as _migrate
        n = await _migrate()
        ok(f"Migrations: {n} applied")
        return True
    except Exception as exc:
        err(f"Migration failed: {exc}")
        return False
    finally:
        from ebrain.db import close_pool
        await close_pool()


async def smoke_test(database_url: str, qdrant_host: str, qdrant_port: int, model: str) -> bool:
    """Quick end-to-end: record turn → extract memory → recall."""
    import ebrain.db as db
    db.DATABASE_URL = database_url
    db._pool = None

    os.environ["EBRAIN_QDRANT_HOST"] = qdrant_host
    os.environ["EBRAIN_QDRANT_PORT"] = str(qdrant_port)

    configure_llm(model)

    try:
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.pipeline import MemoryPipeline

        config = MemoryConfig(l1_every_n_conversations=1)
        async with MemoryPipeline(config=config) as pipeline:
            sid = "ebrain-setup-smoke"
            await pipeline.record(sid, "user", "My name is Alex. I prefer dark mode and use Python.")
            await pipeline.record(sid, "assistant", "Got it, Alex. Dark mode and Python noted.")
            await pipeline.record(sid, "user", "I work on AI agents and memory systems.")

            result = await pipeline.run_pipeline(sid)
            ok(f"Pipeline: L1={result['l1_extracted']} memories extracted")

            recall_result = await pipeline.recall("Python preferences", sid)
            ok(f"Recall: {len(recall_result.memories)} memories found")

        return True
    except Exception as exc:
        err(f"Smoke test failed: {exc}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        from ebrain.db import close_pool
        await close_pool()


# ── Interactive helpers ────────────────────────────────────────────────────────

def prompt_default(question: str, default: str) -> str:
    answer = input(f"  {question} [{default}]: ").strip()
    return answer if answer else default


def prompt_yn(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"  {question} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


# ── Main wizard ───────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(description="EBrain Setup Wizard")
    parser.add_argument("--vault", help="Obsidian vault path (optional)")
    parser.add_argument("--non-interactive", action="store_true", help="Use all defaults")
    args = parser.parse_args()
    non_interactive = args.non_interactive

    print(f"\n{BOLD}{'═' * 55}{RESET}")
    print(f"{BOLD}  EBRAIN SETUP WIZARD{RESET}")
    print(f"{BOLD}  Standalone agent memory + knowledge graph{RESET}")
    print(f"{BOLD}{'═' * 55}{RESET}\n")

    # ── Step 1: Prerequisites ──────────────────────────────────────────────────
    header("Step 1/6 — Prerequisites")
    if not check_python():
        return 1
    has_docker = check_docker()
    has_llm, llm_model = check_opencode()
    if not has_llm:
        return 1
    if not install_ebrain():
        return 1

    # ── Step 2: Database ───────────────────────────────────────────────────────
    header("Step 2/6 — PostgreSQL")
    if has_docker:
        if non_interactive or prompt_yn("Start ebrain-db container?", default=True):
            pg_port = 5432
            if not non_interactive:
                pg_port = int(prompt_default("PostgreSQL port", "5432"))
            if not start_postgres(port=pg_port):
                if not non_interactive and prompt_yn("Use existing PostgreSQL?", default=True):
                    pg_port = int(prompt_default("Existing PostgreSQL port", "5432"))
                else:
                    return 1
        else:
            pg_port = int(prompt_default("Existing PostgreSQL port", "5432"))
    else:
        warn("Docker not available — using existing PostgreSQL")
        pg_port = int(prompt_default("PostgreSQL port", "5432") if not non_interactive else "5432")

    pg_host = "127.0.0.1"
    pg_user = "ebrain"
    pg_pass = "ebrain"
    pg_db = "ebrain"
    database_url = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    ok(f"DATABASE_URL: {database_url}")

    # ── Step 3: Qdrant ─────────────────────────────────────────────────────────
    header("Step 3/6 — Qdrant (vector search)")
    qdrant_port = 6333
    if has_docker:
        if non_interactive or prompt_yn("Start ebrain-qdrant container?", default=True):
            if not non_interactive:
                qdrant_port = int(prompt_default("Qdrant port", "6333"))
            if not start_qdrant(port=qdrant_port):
                if not non_interactive and prompt_yn("Use existing Qdrant?", default=True):
                    qdrant_port = int(prompt_default("Existing Qdrant port", "6333"))
                else:
                    return 1
        else:
            qdrant_port = int(prompt_default("Existing Qdrant port", "6333"))
    else:
        qdrant_port = int(prompt_default("Qdrant port", "6333") if not non_interactive else "6333")

    qdrant_host = "127.0.0.1"
    ok(f"Qdrant: {qdrant_host}:{qdrant_port}")

    # ── Step 4: Migrations ─────────────────────────────────────────────────────
    header("Step 4/6 — Database schema")
    if not await run_migrations(database_url):
        return 1

    # ── Step 5: Vault (Obsidian) ───────────────────────────────────────────────
    header("Step 5/6 — Obsidian vault (optional)")
    vault_path = args.vault or ""
    if not vault_path and not non_interactive:
        if prompt_yn("Configure Obsidian vault sync?", default=False):
            vault_path = prompt_default("Vault path", str(Path.home() / "Documents" / "Obsidian"))

    if vault_path:
        v = Path(vault_path).expanduser()
        if v.exists():
            ok(f"Vault: {v}")
        else:
            warn(f"Vault path doesn't exist yet: {v} (will be created on first sync)")

    # ── Step 6: Env + smoke test ───────────────────────────────────────────────
    header("Step 6/6 — Configure & smoke test")

    env_config = {
        "EBRAIN_DATABASE_URL": database_url,
        "EBRAIN_QDRANT_HOST": qdrant_host,
        "EBRAIN_QDRANT_PORT": str(qdrant_port),
        "EBRAIN_LLM_MODEL": llm_model,
    }
    if vault_path:
        env_config["EBRAIN_VAULT_PATH"] = str(Path(vault_path).expanduser())

    write_env(env_config)

    info("Running smoke test (L0 → L1 → recall)...")
    if not await smoke_test(database_url, qdrant_host, qdrant_port, llm_model):
        err("Smoke test failed. Check logs above.")
        return 1

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 55}{RESET}")
    print(f"{GREEN}{BOLD}  EBRAIN READY!{RESET}")
    print(f"{BOLD}{'═' * 55}{RESET}\n")
    print(f"  LLM backend:  {llm_model}")
    print(f"  Database:     {database_url}")
    print(f"  Qdrant:       {qdrant_host}:{qdrant_port}")
    if vault_path:
        print(f"  Vault:        {vault_path}")
    print(f"\n  Config:  source {ENV_FILE}")
    print("\n  Test:    ebrain stats")
    print("  Recall:  ebrain recall 'python preferences'")
    if vault_path:
        print(f"  Sync:    ebrain vault sync --vault {vault_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
