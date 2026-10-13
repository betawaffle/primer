"""The run explorer's server (tools/viz): reading run directories and serving them."""
import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

VIZ = Path(__file__).resolve().parents[1] / "tools" / "viz"


def load(name):
    spec = importlib.util.spec_from_file_location(f"viz_{name}", VIZ / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


server = load("server")
fake_run = load("fake_run")


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        fake_run.main([str(root / "new"), "--steps", "1000", "--eval", "250"])
        old = root / "old"            # the format from before primer#21: no run.json, "train" is the batch average
        old.mkdir()
        lines = [{"step": s, "train": {"loss": 2.0 / s}, "tokens": s, "seconds": s,
                  "test": {"loss": 3.0, "exact_acc": 0.4}} for s in (100, 200)]
        (old / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines) + '{"step": 300, "tra')
        (root / "not-a-run").mkdir()
        server.Handler.runs = server.Runs([root])
        cls.httpd = server.Server(("127.0.0.1", 0), server.Handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return r.read(), r.headers.get("Content-Type")

    def test_lists_runs(self):
        runs = json.loads(self.get("/api/runs")[0])
        self.assertEqual({r["name"] for r in runs}, {"new", "old"})
        new = next(r for r in runs if r["name"] == "new")
        self.assertEqual((new["step"], new["steps"]), (1000, 1000))
        self.assertFalse(new["live"])                    # finished, however recently written
        self.assertIn("exact_acc", new["latest"]["test"])

    def test_run_entries_and_since(self):
        r = json.loads(self.get("/api/runs/new")[0])
        self.assertEqual([e["step"] for e in r["entries"]], [250, 500, 750, 1000])
        self.assertEqual(len(r["checkpoints"]), 4)
        self.assertIn("test_by", r["entries"][0])
        r = json.loads(self.get("/api/runs/new?since=500")[0])
        self.assertEqual([e["step"] for e in r["entries"]], [750, 1000])

    def test_old_format_and_partial_line(self):
        r = json.loads(self.get("/api/runs/old")[0])
        self.assertIsNone(r["meta"])
        self.assertEqual([e["step"] for e in r["entries"]], [100, 200])     # the half-written line is skipped
        self.assertIn("train_avg", r["entries"][0])
        self.assertNotIn("train", r["entries"][0])

    def test_unknown_run_and_traversal(self):
        for path in ("/api/runs/nope", "/api/runs/..%2F..%2Fetc", "/api/runs/not-a-run"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.get(path)
            self.assertEqual(cm.exception.code, 404)
        try:
            body = self.get("/../../../etc/passwd")[0]
        except urllib.error.HTTPError as e:
            body = e.read()
        self.assertNotIn(b"root:", body)

    def test_static_page(self):
        body, kind = self.get("/")
        self.assertIn(b"Run explorer", body)
        self.assertTrue(kind.startswith("text/html"))

    def test_nan_becomes_null(self):
        self.assertEqual(server.clean({"a": [float("nan"), 1.0, float("inf")]}), {"a": [None, 1.0, None]})


if __name__ == "__main__":
    unittest.main()
