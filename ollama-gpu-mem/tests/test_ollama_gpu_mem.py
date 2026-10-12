import io
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama_gpu_mem as ogm  # noqa: E402

GiB = 1024**3

PS_PAYLOAD = {
    "models": [
        {"name": "llama3:8b", "size": 6 * GiB, "size_vram": 6 * GiB, "context_length": 8192},
        {"name": "mixtral:8x7b", "size": 30 * GiB, "size_vram": 18 * GiB},
        {"name": "phi3:mini", "size": 3 * GiB, "size_vram": 0},
    ]
}


class ModelMemTests(unittest.TestCase):
    def test_processor_matches_ollama_ps(self):
        full, split, cpu = ogm.parse_ps(PS_PAYLOAD)
        self.assertEqual(full.processor, "100% GPU")
        self.assertEqual(split.processor, "40%/60% CPU/GPU")
        self.assertEqual(cpu.processor, "100% CPU")
        self.assertEqual(split.size_ram, 12 * GiB)

    def test_missing_size_vram_and_zero_size(self):
        (m,) = ogm.parse_ps({"models": [{"model": "old:latest", "size": 0}]})
        self.assertEqual(m.name, "old:latest")
        self.assertEqual(m.gpu_percent, 0)
        self.assertEqual(m.processor, "100% CPU")

    def test_empty_payload(self):
        self.assertEqual(ogm.parse_ps({"models": None}), [])
        self.assertEqual(ogm.parse_ps({}), [])


class ParsingTests(unittest.TestCase):
    def test_nvidia_smi_parse_handles_commas_and_na(self):
        out = "0, NVIDIA GeForce RTX 4090, 18432, 24564\n1, Some, GPU, 100, 200\n2, X, [N/A], [N/A]\n"
        gpus = ogm.parse_nvidia_smi(out)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0].used, 18432 * 1024**2)
        self.assertEqual(gpus[1].name, "Some, GPU")

    def test_compute_apps_only_counts_ollama(self):
        out = (
            "1234, /usr/local/bin/ollama, 6144\n"
            "1300, /usr/lib/ollama/runners/cuda_v12/ollama_llama_server, 2048\n"
            "999, /usr/bin/Xorg, 300\n"
            "1400, ollama, [N/A]\n"
        )
        self.assertEqual(ogm.parse_compute_apps(out), 8192 * 1024**2)

    def test_fmt_bytes_uses_binary_units(self):
        self.assertEqual(ogm.fmt_bytes(512), "512 B")
        self.assertEqual(ogm.fmt_bytes(1536 * 1024**2), "1.5 GiB")
        self.assertEqual(ogm.fmt_bytes(4 * 1024**4), "4096.0 GiB")

    def test_normalize_host(self):
        cases = {
            "0.0.0.0": "http://127.0.0.1:11434",
            "0.0.0.0:11434": "http://127.0.0.1:11434",
            "gpu-box": "http://gpu-box:11434",
            "gpu-box:8080/": "http://gpu-box:8080",
            "https://ollama.example.com": "https://ollama.example.com:443",
            "http://[::1]:11434": "http://[::1]:11434",
        }
        for given, want in cases.items():
            with self.subTest(given=given):
                self.assertEqual(ogm.normalize_host(given), want)


class MismatchTests(unittest.TestCase):
    def model(self, vram_gib):
        return ogm.ModelMem("m", int(vram_gib * GiB), int(vram_gib * GiB))

    def test_normal_overhead_is_quiet(self):
        # Real readings: qwen3.8 at 64k (16.3 vs 19.9), qwen3-coder (20.5 vs 20.7).
        self.assertIsNone(ogm.vram_mismatch([self.model(16.3)], int(19.9 * GiB)))
        self.assertIsNone(ogm.vram_mismatch([self.model(20.5)], int(20.7 * GiB)))

    def test_under_reported_model_warns(self):
        # gemma4:31b reported 1.6 GiB while its runner held 22 GiB.
        msg = ogm.vram_mismatch([self.model(1.6)], 22 * GiB)
        self.assertIn("22.0 GiB", msg)
        self.assertIn("20.4 GiB unaccounted", msg)

    def test_stale_runner_with_nothing_listed_warns(self):
        self.assertIsNotNone(ogm.vram_mismatch([], 10 * GiB))

    def test_unknown_or_invisible_processes_are_quiet(self):
        self.assertIsNone(ogm.vram_mismatch([self.model(1.6)], None))
        self.assertIsNone(ogm.vram_mismatch([self.model(18)], 0))

    def test_render_includes_warning(self):
        out = ogm.render([self.model(1.6)], [], 22 * GiB)
        self.assertIn("WARNING", out)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(PS_PAYLOAD).encode() if self.path == "/api/ps" else b"{}"
                self.send_response(200 if self.path == "/api/ps" else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.host = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

        # Fake nvidia-smi on PATH that answers both query modes.
        cls.bindir = tempfile.mkdtemp()
        script = Path(cls.bindir) / "nvidia-smi"
        script.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            '  --query-gpu=*) echo "0, NVIDIA GeForce RTX 4090, 24800, 24564";;\n'
            '  --query-compute-apps=*) echo "42, /usr/bin/ollama, 24576";;\n'
            "esac\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        cls.old_path = os.environ["PATH"]
        os.environ["PATH"] = cls.bindir + os.pathsep + cls.old_path

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        os.environ["PATH"] = cls.old_path

    def run_main(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ogm.main(["--host", self.host, *args])
        return rc, buf.getvalue()

    def test_table_output(self):
        rc, out = self.run_main()
        self.assertEqual(rc, 0)
        self.assertIn("mixtral:8x7b", out)
        self.assertIn("40%/60% CPU/GPU", out)
        self.assertIn("Ollama VRAM in use: 24.0 GiB", out)
        self.assertIn("nvidia-smi, ollama processes: 24.0 GiB", out)
        self.assertIn("GPU 0 NVIDIA GeForce RTX 4090", out)
        self.assertNotIn("WARNING", out)

    def test_json_output(self):
        rc, out = self.run_main("--json")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual([m["gpu_percent"] for m in data["models"]], [100, 60, 0])
        self.assertEqual(data["ollama_process_vram"], 24 * GiB)
        self.assertIsNone(data["warning"])

    def test_unreachable_host(self):
        err = io.StringIO()
        from contextlib import redirect_stderr

        with redirect_stderr(err):
            rc = ogm.main(["--host", "http://127.0.0.1:1", "--no-gpu"])
        self.assertEqual(rc, 1)
        self.assertIn("Could not reach Ollama", err.getvalue())


if __name__ == "__main__":
    unittest.main()
