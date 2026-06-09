"""
Logging and runtime utilities for the pygenebasis CLI.

Adapted from SPIDA settings.py — detects Slurm vs interactive runtime and
configures logging handlers accordingly.
"""
from __future__ import annotations

import logging
import os
import sys

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    from rich.logging import RichHandler  # type: ignore
    _RICH_AVAILABLE = True
except Exception:
    RichHandler = None  # type: ignore
    _RICH_AVAILABLE = False


def detect_runtime() -> str:
    """Detect the runtime environment.

    Returns one of:
      ``'slurm-batch'``       — inside a Slurm batch job (no TTY)
      ``'slurm-interactive'`` — inside Slurm but interactive (srun/salloc or TTY)
      ``'interactive'``       — not under Slurm
    """
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_NTASKS",
        "SLURM_PROCID",
        "SLURM_ARRAY_TASK_ID",
    )
    if not any(k in os.environ for k in slurm_keys):
        return "interactive"

    try:
        if sys.stdout.isatty() or sys.stderr.isatty():
            return "slurm-interactive"
    except Exception:
        pass

    parent_cmd = _get_parent_cmdline().lower()
    if any(x in parent_cmd for x in ("srun", "salloc")):
        return "slurm-interactive"
    if "sbatch" in parent_cmd:
        return "slurm-batch"

    return "slurm-batch"


def _get_parent_cmdline() -> str:
    """Return parent process command line as a string, best-effort."""
    try:
        ppid = os.getppid()
    except Exception:
        return ""

    if psutil:
        try:
            return " ".join(psutil.Process(ppid).cmdline())
        except Exception:
            pass

    try:
        with open(f"/proc/{ppid}/cmdline", "rb") as fh:
            data = fh.read()
            parts = [p.decode("utf-8", errors="replace") for p in data.split(b"\x00") if p]
            return " ".join(parts)
    except Exception:
        return ""


def configure_logging(
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
    log_dir: str | None = None,
) -> str:
    """Configure logging for the detected runtime and return the environment string.

    Behaviors:
    * ``interactive``        — RichHandler (if available) else StreamHandler(stdout)
    * ``slurm-interactive``  — RichHandler with markup=False, else plain StreamHandler
    * ``slurm-batch``        — StreamHandler(stdout) so Slurm's output file captures logs;
                               optionally also a FileHandler in *log_dir*.

    Returns
    -------
    str
        Detected environment (``'interactive'``, ``'slurm-interactive'``,
        or ``'slurm-batch'``).
    """
    env = detect_runtime()
    root = logger if logger is not None else logging.getLogger()

    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass

    fmt = "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s - %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"
    formatter = logging.Formatter(fmt, datefmt)

    if env == "interactive":
        if _RICH_AVAILABLE and RichHandler is not None:
            rh = RichHandler(rich_tracebacks=True, show_time=False, markup=True)
            rh.setLevel(level)
            root.addHandler(rh)
        else:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(level)
            sh.setFormatter(formatter)
            root.addHandler(sh)

    elif env == "slurm-interactive":
        if _RICH_AVAILABLE and RichHandler is not None:
            try:
                rh = RichHandler(rich_tracebacks=False, show_time=False, markup=False)
                rh.setLevel(level)
                root.addHandler(rh)
            except Exception:
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(level)
                sh.setFormatter(formatter)
                root.addHandler(sh)
        else:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(level)
            sh.setFormatter(formatter)
            root.addHandler(sh)

    else:  # slurm-batch
        if log_dir:
            try:
                jobid = os.environ.get("SLURM_JOB_ID", "unknown")
                os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(os.path.join(log_dir, f"slurm_{jobid}.log"))
                fh.setLevel(level)
                fh.setFormatter(formatter)
                root.addHandler(fh)
            except Exception:
                pass

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    root.setLevel(level)
    return env
