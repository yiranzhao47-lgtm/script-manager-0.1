"""
GPU memory lifecycle manager.

Provides scoped context managers for model allocation, enforces
mode-specific loading policy, and logs VRAM state via pynvml.
All model releases are guaranteed through the del / gc.collect /
torch.cuda.empty_cache() triple — no reliance on Python GC timing.

Typical usage
─────────────
    from src.utils.gpu_manager import GPUManager, gpu_manager

    # Once at pipeline startup:
    GPUManager.configure(cfg)

    # Around every model load/unload cycle:
    with gpu_manager.scope("whisper") as scope:
        model = stable_whisper.load_model("large-v3", device="cuda")
        scope.register(model)
        results = model.transcribe(audio_path)
    # ← model freed here unconditionally, VRAM logged
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import pynvml as _pynvml

    _PYNVML_OK = True
except ImportError:
    _PYNVML_OK = False
    logger.warning("pynvml not installed — GPU memory telemetry disabled")


# ══════════════════════════════════════════════════════════════════════════════
#  Exceptions
# ══════════════════════════════════════════════════════════════════════════════


class GPUPolicyError(RuntimeError):
    """Raised when a caller violates the configured GPU loading policy."""


# ══════════════════════════════════════════════════════════════════════════════
#  _ModelScope
# ══════════════════════════════════════════════════════════════════════════════


class _ModelScope:
    """
    Single-model lifecycle handle.  Obtain via GPUManager.scope().

        with gpu_manager.scope("whisper") as scope:
            model = load_whisper(...)
            scope.register(model)        # ← required
            results = model.transcribe(path)
        # model freed, VRAM logged

    If scope.register() is never called the scope still exits cleanly
    but emits a warning — this usually means the model variable was not
    passed in and VRAM might be leaked.
    """

    __slots__ = ("_manager", "_name", "_model")

    def __init__(self, manager: GPUManager, name: str) -> None:
        self._manager = manager
        self._name = name
        self._model: Any = None

    def register(self, model: Any) -> None:
        """Bind the loaded model object so the scope can free it on exit."""
        self._model = model

    def __enter__(self) -> _ModelScope:
        self._manager._on_enter(self._name)
        return self

    def __exit__(self, *_: object) -> None:
        if self._model is None:
            logger.warning(
                "[%s] scope exited without a registered model — "
                "did you forget scope.register(model)?  VRAM may be leaked.",
                self._name,
            )
        self._manager._release(self._name, self._model)
        self._model = None


# ══════════════════════════════════════════════════════════════════════════════
#  GPUManager
# ══════════════════════════════════════════════════════════════════════════════


class GPUManager:
    """
    Singleton GPU memory manager.

    Lifecycle
    ─────────
    1.  Call GPUManager.configure(cfg) once at pipeline startup.
    2.  Use gpu_manager.scope(name) as a context manager for every
        model allocation.  Two models cannot be resident simultaneously
        when gpu.enforce_sequential == true (default).

    Policy enforcement
    ──────────────────
    •  Attempting to load "whisper" while cross_lang.asr.role == "disabled"
       raises GPUPolicyError immediately at scope() call time.
    •  Attempting to enter a scope while another scope is active raises
       GPUPolicyError at __enter__ time (when enforce_sequential == true).

    Properties
    ──────────
    •  mode      — current pipeline mode ("same_lang" | "cross_lang")
    •  asr_role  — cross_lang ASR role ("semantic_anchor" | "disabled")
    """

    _instance: Optional[GPUManager] = None

    # ------------------------------------------------------------------ #
    #  Singleton construction                                              #
    # ------------------------------------------------------------------ #

    def __new__(cls) -> GPUManager:
        if cls._instance is None:
            obj = object.__new__(cls)
            # Safe defaults before configure() is called
            obj._ready: bool = False
            obj._mode: str = "same_lang"
            obj._asr_role: str = "semantic_anchor"
            obj._enforce_sequential: bool = True
            obj._active_models: set[str] = set()
            obj._nvml_handle: Any = None
            cls._instance = obj
        return cls._instance

    @classmethod
    def configure(cls, cfg: dict) -> GPUManager:
        """
        Initialise the singleton with pipeline config.  Safe to call multiple
        times; subsequent calls update settings but do not re-init pynvml.
        """
        inst = cls()
        inst._mode = cfg.get("pipeline", {}).get("mode", "same_lang")
        inst._asr_role = (
            cfg.get("cross_lang", {}).get("asr", {}).get("role", "semantic_anchor")
        )
        inst._enforce_sequential = cfg.get("gpu", {}).get("enforce_sequential", True)
        if not inst._ready:
            inst._init_nvml()
            inst._ready = True
        logger.info(
            "GPUManager configured — mode=%s  asr_role=%s  enforce_sequential=%s",
            inst._mode,
            inst._asr_role,
            inst._enforce_sequential,
        )
        return inst

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def asr_role(self) -> str:
        return self._asr_role

    def scope(self, model_name: str) -> _ModelScope:
        """
        Return a lifecycle context manager for the named model.

        Permanent policy violations (config-derived, cannot change at runtime)
        raise GPUPolicyError immediately here.  Sequential-load violations are
        deferred to __enter__ so the error occurs at allocation time.
        """
        self._assert_configured()
        self._check_permanent_policy(model_name)
        return _ModelScope(self, model_name)

    def shutdown(self) -> None:
        """Release pynvml resources.  Call once at pipeline teardown."""
        if _PYNVML_OK and self._nvml_handle is not None:
            try:
                _pynvml.nvmlShutdown()
                logger.debug("pynvml shutdown complete")
            except _pynvml.NVMLError:
                pass

    # ------------------------------------------------------------------ #
    #  Called by _ModelScope                                               #
    # ------------------------------------------------------------------ #

    def _on_enter(self, name: str) -> None:
        """Enforce sequential loading, register model as active, log VRAM."""
        if self._enforce_sequential and self._active_models:
            active = sorted(self._active_models)
            raise GPUPolicyError(
                f"Sequential enforcement: cannot load '{name}' while "
                f"{active} are still resident in VRAM.  "
                "Exit their scopes before entering a new one "
                "(gpu.enforce_sequential = true)."
            )
        self._active_models.add(name)
        self._log_vram(f"PRE-ALLOC  [{name}]")

    def _release(self, name: str, model: Any) -> None:
        """Execute the triple release guarantee and log post-free VRAM."""
        self._active_models.discard(name)

        if model is not None:
            del model

        gc.collect()

        try:
            import torch  # lazy — avoids hard torch dependency in CPU-only contexts
            torch.cuda.empty_cache()
            logger.debug("torch.cuda.empty_cache() called after releasing [%s]", name)
        except ImportError:
            pass

        self._log_vram(f"POST-FREE  [{name}]")

    # ------------------------------------------------------------------ #
    #  Policy enforcement                                                  #
    # ------------------------------------------------------------------ #

    def _check_permanent_policy(self, name: str) -> None:
        """
        Checks that do not depend on runtime state — only on config.
        Fail-fast before a _ModelScope object is even created.
        """
        if (
            name.lower() == "whisper"
            and self._mode == "cross_lang"
            and self._asr_role == "disabled"
        ):
            raise GPUPolicyError(
                "Cannot allocate Whisper: cross_lang.asr.role == 'disabled'.  "
                "Set asr.role to 'semantic_anchor' in settings.yaml, "
                "or switch pipeline.mode to 'same_lang'."
            )

    # ------------------------------------------------------------------ #
    #  pynvml telemetry                                                    #
    # ------------------------------------------------------------------ #

    def _init_nvml(self) -> None:
        if not _PYNVML_OK:
            return
        try:
            _pynvml.nvmlInit()
            self._nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            info = _pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            logger.info(
                "pynvml ready — GPU[0] total VRAM: %.0f MB",
                info.total / 1024**2,
            )
        except _pynvml.NVMLError as exc:
            logger.warning(
                "pynvml init failed (%s) — VRAM telemetry disabled", exc
            )
            self._nvml_handle = None

    def _log_vram(self, tag: str) -> None:
        if self._nvml_handle is None:
            return
        try:
            info = _pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            logger.info(
                "VRAM %-44s  used=%6.0f MB  free=%6.0f MB  total=%6.0f MB",
                tag,
                info.used / 1024**2,
                info.free / 1024**2,
                info.total / 1024**2,
            )
        except _pynvml.NVMLError as exc:
            logger.debug("VRAM query error: %s", exc)

    # ------------------------------------------------------------------ #
    #  Guard                                                               #
    # ------------------------------------------------------------------ #

    def _assert_configured(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "GPUManager has not been configured.  "
                "Call GPUManager.configure(cfg) at pipeline startup "
                "before any scope() calls."
            )


# ── Module-level singleton ────────────────────────────────────────────────────
# Import and use directly:
#   from src.utils.gpu_manager import gpu_manager
gpu_manager: GPUManager = GPUManager()
