import io
import itertools
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ollama_bench as ob  # noqa: E402

GOOD_CODE = '''```python
import re

def parse_duration(s: str) -> int:
    m = re.fullmatch(r"(?:(\\d+)d)?(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", s)
    if not s or not m:
        raise ValueError(s)
    d, h, mi, se = (int(g or 0) for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se
```'''

# Accepts "1h 30m" by stripping spaces: should fail the tests.
SLOPPY_CODE = GOOD_CODE.replace(
    "def parse_duration(s: str) -> int:\n", "def parse_duration(s: str) -> int:\n    s = s.replace(' ', '')\n")


class AnswerKeyTests(unittest.TestCase):
    """Verify the expected answers independently, by brute force."""

    def test_conveyor_answer(self):
        boxes, t, dt = 0.0, 0.0, 0.001
        while boxes < 1000:
            rate = (12 if not 20 <= t < 27 else 0) + (18 if t >= 10 else 0) + (25 if t >= 15 else 0)
            boxes += rate * dt
            t += dt
        self.assertAlmostEqual(t, 29.8, places=1)

    def test_schedule_answer(self):
        dur = {"A": 3, "B": 2, "C": 4, "D": 2, "E": 3, "F": 5}
        deps = {"C": "A", "D": "AB", "E": "CD"}
        best = None
        # Every task order x every worker assignment; greedy earliest start.
        for order in itertools.permutations(dur):
            if any(order.index(p) > order.index(t) for t, ps in deps.items() for p in ps):
                continue
            for assign in itertools.product((0, 1), repeat=6):
                free, end = [0, 0], {}
                for task, w in zip(order, assign):
                    start = max([free[w]] + [end[p] for p in deps.get(task, "")])
                    end[task] = free[w] = start + dur[task]
                span = max(end.values())
                best = span if best is None else min(best, span)
        self.assertEqual(best, 10)


class CheckerTests(unittest.TestCase):
    def test_code_good_and_bad(self):
        self.assertEqual(ob.check_code(GOOD_CODE).status, "PASS")
        self.assertEqual(ob.check_code(SLOPPY_CODE).status, "FAIL")
        self.assertEqual(ob.check_code("no code here").status, "FAIL")
        for label in ("Python", "python3", ""):
            with self.subTest(label=label):
                self.assertEqual(ob.check_code(GOOD_CODE.replace("```python", "```" + label)).status, "PASS")

    def test_code_from_saved_answer_file(self):
        # The saved .md holds the prompt (which mentions ```python inline) before the answer.
        saved = f"## Prompt\n\n{ob.CODE_PROMPT}\n\n## Answer\n\n"
        self.assertEqual(ob.check_code(saved + GOOD_CODE).status, "PASS")
        self.assertEqual(ob.check_code(saved).status, "FAIL")
        self.assertEqual(ob.check_code("```python\nwhile True: pass\n```").status, "FAIL")

    def test_code_failure_names_the_input(self):
        rejects_zero = GOOD_CODE.replace("    return d *", "    if d + h + mi + se == 0:\n        raise ValueError\n    return d *")
        res = ob.check_code(rejects_zero)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("'0s' raised ValueError, want 0", res.detail)
        self.assertIn("1/19 cases failed", res.detail)

    def test_number_answers(self):
        check = ob.check_number(29.8)
        self.assertEqual(check("work...\nANSWER: 29.8").status, "PASS")
        self.assertEqual(check("**ANSWER:** 29.8").status, "PASS")
        self.assertEqual(check("ANSWER: 31\n...\nANSWER: 29.8.").status, "PASS")
        self.assertEqual(check("ANSWER: 30.2").status, "FAIL")
        self.assertEqual(check("the answer is 29.8").status, "FAIL")

    def test_bug_line(self):
        self.assertEqual(ob.check_bug("Off by one.\nLINE: 8").status, "PASS")
        self.assertEqual(ob.check_bug("LINE: 10").status, "FAIL")

    def test_format(self):
        good = "- Anyone on the network can use your GPU\n- Prompts may leak\n- Models can be deleted"
        self.assertEqual(ob.check_format(good).status, "PASS")
        self.assertEqual(ob.check_format("Here you go:\n" + good).status, "FAIL")
        self.assertEqual(ob.check_format(good.replace("leak", "leak " + "word " * 12)).status, "FAIL")

    def test_thinking_is_stripped(self):
        self.assertEqual(ob.strip_thinking("<think>hmm\nANSWER: 1</think>ANSWER: 10"), "ANSWER: 10")


class EndToEndTests(unittest.TestCase):
    def test_run_against_fake_ollama(self):
        replies = {"code": GOOD_CODE, "bug": "LINE: 8", "math": "ANSWER: 29.8",
                   "schedule": "ANSWER: 12", "format": "- a\n- b\n- c", "design": "word " * 300}
        seen_options = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                task = next(t for t, (_, p, _) in ob.TASKS.items() if p == body["prompt"])
                seen_options.append(body["options"])
                reply = {"response": replies[task], "eval_count": 100, "eval_duration": 5_000_000_000}
                if body["model"] == "thinker":
                    reply.update(response="", thinking="hmm " * 50, done_reason="length")
                out = json.dumps(reply).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as d:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = ob.main(["m1", "thinker", "--host", f"http://127.0.0.1:{server.server_port}", "--out", d])
            server.shutdown()
            server.server_close()
            out = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("4/5", out)  # schedule wrong, design not auto-graded
            self.assertIn("20.0", out)  # 100 tokens / 5 s
            self.assertIn("over 250", out)
            self.assertTrue((Path(d) / "m1" / "code.md").exists())
            self.assertIn("hit the 8192-token limit", out)
            self.assertIn("## Thinking (50 words)", (Path(d) / "thinker" / "math.md").read_text())
            self.assertTrue(all("temperature" not in o for o in seen_options))
            self.assertTrue(all(o["seed"] == 42 for o in seen_options))


if __name__ == "__main__":
    unittest.main()
