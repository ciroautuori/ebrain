"""Memory pipeline configuration — TencentDB-inspired layering settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for the 4-layer memory pipeline.

    All layers can be independently enabled/disabled.
    Extraction uses LLM; recall uses vector search (Qdrant).
    """

    # ── L0: Conversation recording ──────────────────────────────
    l0_enabled: bool = True
    l0_retention_days: int = 90  # 0 = no cleanup

    # ── L1: Memory extraction ───────────────────────────────────
    l1_enabled: bool = True
    l1_every_n_conversations: int = 5  # trigger L1 after N convos
    l1_max_memories_per_run: int = 20
    l1_dedup_threshold: float = 0.85  # cosine similarity for dedup
    l1_model: str | None = None  # None = use default (model_for_tier)

    # ── L2: Scene/profile building ──────────────────────────────
    l2_enabled: bool = True
    l2_trigger_every_n_memories: int = 50  # trigger L2 after N new L1
    l2_max_scenes: int = 15
    l2_model: str | None = None

    # ── L3: Persona generation ──────────────────────────────────
    l3_enabled: bool = True
    l3_trigger_every_n_scenes: int = 10
    l3_model: str | None = None

    # ── Recall ──────────────────────────────────────────────────
    recall_enabled: bool = True
    recall_max_results: int = 5
    recall_score_threshold: float = 0.3  # min similarity
    recall_max_chars_per_memory: int = 500  # 0 = unlimited

    # ── Pipeline scheduling ─────────────────────────────────────
    l1_idle_timeout_seconds: float = 600  # trigger L1 after inactivity
    l2_delay_after_l1_seconds: float = 10
    warmup_enabled: bool = True  # accelerate early extraction

    @classmethod
    def from_env(cls) -> MemoryConfig:
        """Build config from EROS_MEMORY_* environment variables (optional)."""
        import os

        def _bool(k: str, default: bool) -> bool:
            v = os.environ.get(k, "").lower()
            if v in ("1", "true", "yes"):
                return True
            if v in ("0", "false", "no"):
                return False
            return default

        def _int(k: str, default: int) -> int:
            try:
                return int(os.environ[k])
            except (KeyError, ValueError):
                return default

        def _float(k: str, default: float) -> float:
            try:
                return float(os.environ[k])
            except (KeyError, ValueError):
                return default

        return cls(
            l0_enabled=_bool("EROS_MEMORY_L0_ENABLED", True),
            l0_retention_days=_int("EROS_MEMORY_L0_RETENTION_DAYS", 90),
            l1_enabled=_bool("EROS_MEMORY_L1_ENABLED", True),
            l1_every_n_conversations=_int("EROS_MEMORY_L1_EVERY_N", 5),
            l1_max_memories_per_run=_int("EROS_MEMORY_L1_MAX_PER_RUN", 20),
            l1_dedup_threshold=_float("EROS_MEMORY_L1_DEDUP_THRESHOLD", 0.85),
            l2_enabled=_bool("EROS_MEMORY_L2_ENABLED", True),
            l2_trigger_every_n_memories=_int("EROS_MEMORY_L2_EVERY_N", 50),
            l2_max_scenes=_int("EROS_MEMORY_L2_MAX_SCENES", 15),
            l3_enabled=_bool("EROS_MEMORY_L3_ENABLED", True),
            l3_trigger_every_n_scenes=_int("EROS_MEMORY_L3_EVERY_N", 10),
            recall_enabled=_bool("EROS_MEMORY_RECALL_ENABLED", True),
            recall_max_results=_int("EROS_MEMORY_RECALL_MAX_RESULTS", 5),
            recall_score_threshold=_float("EROS_MEMORY_RECALL_SCORE_THRESHOLD", 0.3),
        )


DEFAULT_CONFIG = MemoryConfig()
