#!/usr/bin/env python3
"""Run a small capability benchmark against local Ollama models.

Five short tasks, each aimed at a different skill. Most are checked
automatically; the design task is saved for you (or a stronger model) to grade.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if "://" not in DEFAULT_HOST:
    DEFAULT_HOST = "http://" + DEFAULT_HOST.replace("0.0.0.0", "127.0.0.1")


# ---------------------------------------------------------------- the tasks

CODE_PROMPT = """Write a Python function `parse_duration(s: str) -> int` that converts a
duration string to a number of seconds.

Rules:
- Units are d (days), h (hours), m (minutes), s (seconds), lowercase only.
- Each unit appears at most once, in the order d, h, m, s. Any may be omitted.
- Each unit is preceded by a non-negative whole number. No spaces, no decimals.
- Values are not normalized: "90m" is valid and means 5400.
- Anything else (empty string, missing unit, unknown unit, wrong order,
  repeated unit, decimals, spaces, uppercase) raises ValueError.

Examples: "1h30m" -> 5400, "2d4h" -> 187200, "45s" -> 45, "0s" -> 0.

Reply with one ```python code block containing only the function (and any
imports it needs). No tests, no example usage."""

CODE_TESTS = """
ok = {"1h30m": 5400, "2d4h": 187200, "45s": 45, "0s": 0, "90m": 5400,
      "1d1h1m1s": 90061, "10d": 864000, "007m": 420}
for s, want in ok.items():
    got = parse_duration(s)
    assert got == want, f"{s!r}: got {got!r}, want {want}"
bad = ["", "10", "h", "1h1h", "30m1h", "1.5h", "1h 30m", "1H", "-5s", "1w", "s1", "1hm"]
for s in bad:
    try:
        parse_duration(s)
    except ValueError:
        continue
    raise AssertionError(f"{s!r} should raise ValueError")
print("ALL TESTS PASSED")
"""

BUG_PROMPT = """This function should return the moving average of `values` over a sliding
window of size `window`. It has exactly one bug.

```
1  def moving_average(values, window):
2      if window <= 0:
3          raise ValueError("window must be positive")
4      result = []
5      total = 0
6      for i, v in enumerate(values):
7          total += v
8          if i > window:
9              total -= values[i - window]
10         if i >= window - 1:
11             result.append(total / window)
12     return result
```

Explain the bug, show the corrected line, and give one input where the original
returns a wrong result. Finish with a final line in exactly this form:
LINE: <line number of the bug>"""

MATH_PROMPT = """Three conveyor belts load boxes into a truck. Belt A moves 12 boxes per minute,
belt B 18 per minute, belt C 25 per minute. Belt A starts at minute 0. Belt B
starts 10 minutes after A. Belt C starts 5 minutes after B. Belt A breaks down
at minute 20 and is repaired 7 minutes later, then continues at its normal
rate. Rates are constant while running.

At what minute (measured from A's start) does the total reach 1000 boxes?
Show your reasoning step by step, then finish with a final line in exactly
this form, rounded to one decimal:
ANSWER: <number>"""

SCHEDULE_PROMPT = """Two workers must complete six tasks. Durations in hours:
A=3, B=2, C=4, D=2, E=3, F=5.
Dependencies: C needs A finished. D needs A and B finished. E needs C and D
finished. F has no dependencies.
A worker does one task at a time, a task cannot be split between workers or
paused, and a task can start only when all its dependencies are finished.

What is the minimum number of hours to finish all six tasks? Give the schedule
(which worker does which task, start and end times), explain why it cannot be
done faster, then finish with a final line in exactly this form:
ANSWER: <number>"""

FORMAT_PROMPT = """List exactly three risks of exposing an Ollama server on 0.0.0.0 without
authentication.

Strict format: exactly three lines, each starting with "- ", each at most 12
words. No heading, no introduction, no blank lines, nothing else."""

DESIGN_PROMPT = """Four people on a home network share one 24 GB GPU running Ollama. Each of
their models needs 18-22 GB, so only one model fits at a time, and switching
models takes about 20 seconds. People send requests for different models at
the same time.

Design a small request broker that sits in front of Ollama. Cover:
1. its components,
2. how it orders requests to reduce model switches while staying fair,
3. what happens to a request that has waited too long,
4. one realistic failure mode and how the broker handles it.

At most 250 words."""


@dataclass
class Result:
    status: str  # PASS / FAIL / REVIEW
    detail: str = ""


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def last_tagged(text: str, tag: str) -> str | None:
    found = re.findall(rf"^\s*\**{tag}\**:\s*\**\s*([^\s*]+)", text, flags=re.M | re.I)
    return found[-1] if found else None


def check_number(want: float) -> Callable[[str], Result]:
    def check(text: str) -> Result:
        got = last_tagged(text, "ANSWER")
        if got is None:
            return Result("FAIL", "no ANSWER line")
        try:
            ok = abs(float(got.rstrip(".").replace(",", "")) - want) < 0.05
        except ValueError:
            return Result("FAIL", f"ANSWER not a number: {got}")
        return Result("PASS" if ok else "FAIL", f"answered {got}, expected {want:g}")

    return check


def check_bug(text: str) -> Result:
    got = last_tagged(text, "LINE")
    if got is None:
        return Result("FAIL", "no LINE line")
    return Result("PASS" if got.rstrip(".") == "8" else "FAIL", f"answered line {got}, expected 8")


def check_format(text: str) -> Result:
    lines = text.strip().splitlines()
    if len(lines) != 3:
        return Result("FAIL", f"{len(lines)} lines, expected 3")
    for ln in lines:
        if not ln.startswith("- "):
            return Result("FAIL", f"line does not start with '- ': {ln[:40]!r}")
        if len(ln[2:].split()) > 12:
            return Result("FAIL", f"line over 12 words: {ln[:40]!r}")
    return Result("PASS", "3 lines, all within limits")


def check_code(text: str) -> Result:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        return Result("FAIL", "no code block")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "solution.py"
        path.write_text(blocks[-1] + "\n" + CODE_TESTS)
        try:
            run = subprocess.run(
                [sys.executable, "-I", str(path)], cwd=d, capture_output=True, text=True, timeout=10
            )
        except subprocess.TimeoutExpired:
            return Result("FAIL", "code timed out")
    if run.returncode == 0 and "ALL TESTS PASSED" in run.stdout:
        return Result("PASS", "all 20 test cases pass")
    err = (run.stderr.strip().splitlines() or ["unknown error"])[-1]
    return Result("FAIL", err[:120])


def check_design(text: str) -> Result:
    words = len(text.split())
    return Result("REVIEW", f"{words} words{' (over 250!)' if words > 250 else ''}")


TASKS: dict[str, tuple[str, str, Callable[[str], Result]]] = {
    # name: (skill, prompt, checker)
    "code": ("coding", CODE_PROMPT, check_code),
    "bug": ("code review", BUG_PROMPT, check_bug),
    "math": ("multi-step arithmetic", MATH_PROMPT, check_number(29.8)),
    "schedule": ("planning / constraints", SCHEDULE_PROMPT, check_number(10)),
    "format": ("instruction following", FORMAT_PROMPT, check_format),
    "design": ("architecture", DESIGN_PROMPT, check_design),
}


# ---------------------------------------------------------------- running

def generate(host: str, model: str, prompt: str, num_predict: int, timeout: float) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def render_summary(models: list[str], tasks: list[str], results: dict, speed: dict) -> str:
    rows = [["TASK", "SKILL", *models]]
    for t in tasks:
        rows.append([t, TASKS[t][0], *(results[m][t].status for m in models)])
    score = [f"{sum(results[m][t].status == 'PASS' for t in tasks)}/"
             f"{sum(results[m][t].status != 'REVIEW' for t in tasks)}" for m in models]
    rows.append(["score", "auto-checked", *score])
    rows.append(["speed", "tokens/s", *(f"{speed[m][0]:.1f}" for m in models)])
    rows.append(["time", "total", *(f"{speed[m][1]:.0f}s" for m in models)])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="models to test, e.g. qwen3-coder-64k:30b")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--tasks", default=",".join(TASKS), help=f"comma list (default: {','.join(TASKS)})")
    ap.add_argument("--out", default="bench-results", help="directory for full answers")
    ap.add_argument("--num-predict", type=int, default=8192, help="max tokens per answer")
    ap.add_argument("--timeout", type=float, default=1200, help="seconds per task")
    args = ap.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        ap.error(f"unknown task(s): {', '.join(unknown)}")

    out = Path(args.out)
    results: dict[str, dict[str, Result]] = {}
    speed: dict[str, tuple[float, float]] = {}
    for model in args.models:  # model by model, so each loads only once
        results[model] = {}
        tok = dur = 0.0
        model_start = time.monotonic()
        for t in tasks:
            skill, prompt, check = TASKS[t]
            print(f"{model}  {t:<9}", end="", flush=True)
            start = time.monotonic()
            try:
                r = generate(args.host, model, prompt, args.num_predict, args.timeout)
            except (urllib.error.URLError, OSError, ValueError) as e:
                res, r = Result("FAIL", f"request failed: {e}"), {}
            else:
                text = strip_thinking(r.get("response", ""))
                res = check(text) if text else Result("FAIL", "empty answer (ran out of tokens?)")
                d = out / safe_name(model)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{t}.md").write_text(
                    f"# {model} / {t} ({skill})\n\nResult: {res.status} - {res.detail}\n\n"
                    f"## Prompt\n\n{prompt}\n\n## Answer\n\n{text}\n"
                )
            elapsed = time.monotonic() - start
            tok += r.get("eval_count", 0)
            dur += (r.get("eval_duration") or 0) / 1e9
            results[model][t] = res
            print(f"{res.status:<7}{elapsed:6.0f}s  {res.detail}")
        speed[model] = (tok / dur if dur else 0.0, time.monotonic() - model_start)
        print()

    print(render_summary(args.models, tasks, results, speed))
    print(f"\nFull answers: {out}/<model>/<task>.md")
    if "design" in tasks:
        print("The design task is not auto-graded: read the answers side by side.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
