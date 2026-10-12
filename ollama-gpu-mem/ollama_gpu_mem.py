#!/usr/bin/env python3
"""Show how much of each loaded Ollama model sits in GPU memory vs system RAM.

Reads Ollama's /api/ps endpoint and, when available, nvidia-smi for whole-GPU
totals. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# nvidia-smi normally shows a bit more than Ollama's estimate (CUDA context,
# compute buffers): up to ~3.5 GiB on a 64k-context model. Only flag gaps
# well beyond that.
MISMATCH_MIN_GAP = 4 * 1024**3
MISMATCH_MIN_RATIO = 1.5


@dataclass
class ModelMem:
    name: str
    size: int  # total bytes the loaded model occupies (weights + KV cache + graph)
    size_vram: int  # bytes of that total resident in GPU memory
    context_length: int | None = None
    expires_at: str | None = None

    @property
    def size_ram(self) -> int:
        return max(self.size - self.size_vram, 0)

    @property
    def gpu_percent(self) -> int:
        if self.size <= 0:
            return 0
        return round(min(self.size_vram, self.size) / self.size * 100)

    @property
    def processor(self) -> str:
        """Same wording as the PROCESSOR column of `ollama ps`."""
        if self.size_vram <= 0:
            return "100% CPU"
        if self.size_vram >= self.size:
            return "100% GPU"
        gpu = self.gpu_percent
        return f"{100 - gpu}%/{gpu}% CPU/GPU"


@dataclass
class GpuMem:
    index: int
    name: str
    used: int  # bytes
    total: int  # bytes


def normalize_host(host: str) -> str:
    """Accept the same forms OLLAMA_HOST does: '0.0.0.0', 'host:port', full URL."""
    host = host.strip().rstrip("/")
    if "://" not in host:
        host = "http://" + host
    u = urllib.parse.urlsplit(host)
    hostname = u.hostname or "127.0.0.1"
    # 0.0.0.0 is a bind address, not something you can connect to.
    if hostname == "0.0.0.0":
        hostname = "127.0.0.1"
    if ":" in hostname:
        hostname = f"[{hostname}]"  # IPv6 literal
    port = u.port or (443 if u.scheme == "https" else 11434)
    return f"{u.scheme}://{hostname}:{port}{u.path}"


def parse_ps(payload: dict) -> list[ModelMem]:
    models = []
    for m in payload.get("models") or []:
        models.append(
            ModelMem(
                name=m.get("name") or m.get("model") or "?",
                size=int(m.get("size") or 0),
                # Older Ollama builds omit size_vram; treat as unknown -> 0.
                size_vram=int(m.get("size_vram") or 0),
                context_length=m.get("context_length"),
                expires_at=m.get("expires_at"),
            )
        )
    return models


def fetch_models(host: str, timeout: float = 5.0) -> list[ModelMem]:
    url = normalize_host(host) + "/api/ps"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return parse_ps(json.load(resp))


def parse_nvidia_smi(output: str) -> list[GpuMem]:
    """Parse `--query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits`.

    nvidia-smi reports memory in MiB.
    """
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx, used, total = int(parts[0]), int(parts[-2]), int(parts[-1])
        except ValueError:
            continue  # "[N/A]" on some boards
        name = ", ".join(parts[1:-2])
        gpus.append(GpuMem(idx, name, used * 1024**2, total * 1024**2))
    return gpus


def fetch_gpus() -> list[GpuMem]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_nvidia_smi(out)


def parse_compute_apps(output: str) -> int:
    """Sum MiB used by Ollama processes from
    `--query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`.
    """
    total = 0
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or "ollama" not in parts[1].lower():
            continue
        try:
            total += int(parts[-1])
        except ValueError:
            continue
    return total * 1024**2


def fetch_ollama_process_vram() -> int | None:
    """VRAM actually held by ollama processes per nvidia-smi, or None if unknown."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return parse_compute_apps(out)


def fmt_bytes(n: int) -> str:
    """Binary units, matching what nvidia-smi and most GPU tools show."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def vram_mismatch(models: list[ModelMem], proc_vram: int | None) -> str | None:
    """Warn when ollama processes hold far more VRAM than Ollama reports.

    Ollama's scheduler decides what fits using its own figures, so a model it
    under-reports (or a stale runner it no longer lists) can lead it to load a
    second model on top and spill to CPU or run out of memory.
    """
    if not proc_vram:  # unknown, or nvidia-smi can't see processes (containers)
        return None
    reported = sum(m.size_vram for m in models)
    gap = proc_vram - reported
    if gap < MISMATCH_MIN_GAP or proc_vram < reported * MISMATCH_MIN_RATIO:
        return None
    return (
        f"WARNING: ollama processes hold {fmt_bytes(proc_vram)} but Ollama reports "
        f"{fmt_bytes(reported)} ({fmt_bytes(gap)} unaccounted). Ollama is "
        "under-reporting a model's size, or a stale runner is holding VRAM. Its "
        "scheduler trusts the reported figure; OLLAMA_MAX_LOADED_MODELS=1 avoids "
        "loading a second model on top."
    )


def render(models: list[ModelMem], gpus: list[GpuMem], proc_vram: int | None = None) -> str:
    lines = []
    if not models:
        lines.append("No models loaded.")
    else:
        rows = [("MODEL", "TOTAL", "VRAM", "RAM", "PROCESSOR", "CTX")]
        for m in models:
            rows.append(
                (
                    m.name,
                    fmt_bytes(m.size),
                    fmt_bytes(m.size_vram),
                    fmt_bytes(m.size_ram),
                    m.processor,
                    str(m.context_length or "-"),
                )
            )
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        for r in rows:
            lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        total_vram = sum(m.size_vram for m in models)
        lines.append(f"\nOllama VRAM in use: {fmt_bytes(total_vram)}")
    if proc_vram is not None:
        lines.append(f"nvidia-smi, ollama processes: {fmt_bytes(proc_vram)}")
    warning = vram_mismatch(models, proc_vram)

    for g in gpus:
        pct = g.used / g.total * 100 if g.total else 0
        lines.append(
            f"GPU {g.index} {g.name}: {fmt_bytes(g.used)} / {fmt_bytes(g.total)} ({pct:.0f}%)"
        )
    if warning:
        lines.append("\n" + warning)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama URL (default: $OLLAMA_HOST or localhost:11434)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--watch", type=float, metavar="SECS", help="refresh every SECS seconds")
    ap.add_argument("--no-gpu", action="store_true", help="skip nvidia-smi")
    args = ap.parse_args(argv)

    while True:
        try:
            models = fetch_models(args.host)
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"Could not reach Ollama at {normalize_host(args.host)}: {e}", file=sys.stderr)
            return 1
        gpus = [] if args.no_gpu else fetch_gpus()
        proc_vram = None if args.no_gpu else fetch_ollama_process_vram()

        if args.json:
            out = {
                "models": [
                    {**asdict(m), "size_ram": m.size_ram, "gpu_percent": m.gpu_percent, "processor": m.processor}
                    for m in models
                ],
                "gpus": [asdict(g) for g in gpus],
                "ollama_process_vram": proc_vram,
                "warning": vram_mismatch(models, proc_vram),
            }
            print(json.dumps(out, indent=2))
        else:
            if args.watch:
                print("\033[2J\033[H", end="")
            print(render(models, gpus, proc_vram))

        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
