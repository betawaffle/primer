"""Run explorer: a small web app for following training runs.

    python3 tools/viz/server.py --runs /data/primer/runs --host 127.0.0.1 --port 8080

Reads run directories as primer.train writes them (run.json, log.jsonl,
checkpoints/step-*.pt) and serves them to the page in static/. Only the
standard library, so it runs anywhere Python 3.11+ does; nothing here imports
torch or touches the GPU. Read-only: it never writes into a run directory.

API (all JSON):
  /api/runs                      every run: name, run.json summary, latest eval, live or not
  /api/runs/<name>?since=<step>  run.json, checkpoint list, and the log entries after <step>
"""
import argparse
import json
import math
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

STATIC = Path(__file__).with_name("static")
LIVE_SECONDS = 600      # a log written to this recently counts as a run in progress


def read_log(path, since=-1):
    """Entries of log.jsonl after step `since`, normalized to the current format.
    Runs from before primer#21 logged the training batches' average as "train"
    and had no evaluation of training sequences; that becomes "train_avg" here.
    A half-written last line (the run is mid-write) is skipped, not an error."""
    out = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict) or e.get("step", -1) <= since:
            continue
        if "train_avg" not in e and isinstance(e.get("train"), dict) and "sequences" not in e["train"]:
            e["train_avg"] = e.pop("train")
        out.append(clean(e))
    return out


def clean(x):
    """JSON has no NaN or infinity; a diverged run would otherwise break the page."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def checkpoints(run):
    d = run / "checkpoints"
    out = []
    for f in sorted(d.glob("step-*.pt")) if d.is_dir() else []:
        try:
            out.append({"step": int(f.stem.split("-")[1]), "bytes": f.stat().st_size})
        except (ValueError, OSError):
            pass
    return out


def last_line(path):
    """The last complete entry of a log, read from its tail so big logs stay cheap."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - (1 << 20)))
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            return e
    return None


SUMMARY = ("exact_acc", "answer_acc", "pointer_acc", "token_acc", "value_acc", "loss")


class Runs:
    def __init__(self, roots):
        self.roots = [Path(r) for r in roots]

    def dirs(self):
        """{name: path} for every directory holding a log.jsonl, one level under each root.
        A name repeated across roots gets the root's name as a prefix."""
        out = {}
        for root in self.roots:
            for d in sorted(root.iterdir()) if root.is_dir() else []:
                if (d / "log.jsonl").is_file():
                    name = d.name if d.name not in out else f"{root.name}/{d.name}"
                    out[name] = d
        return out

    def summary(self, name, d):
        meta = read_json(d / "run.json") or {}
        log = d / "log.jsonl"
        mtime = log.stat().st_mtime
        last = last_line(log) or {}
        if "train_avg" not in last and "train" in last:
            last = {**last, "train_avg": last.pop("train")}
        opts = meta.get("options") or {}
        evals = {k: {m: last[k][m] for m in SUMMARY if isinstance(last.get(k), dict) and m in last[k]}
                 for k in ("test", "val", "train") if isinstance(last.get(k), dict)}
        return clean({"name": name, "started": meta.get("started"), "commit": meta.get("commit"),
                      "dirty": meta.get("dirty"), "steps": opts.get("steps"), "step": last.get("step"),
                      "seconds": last.get("seconds"), "updated": mtime,
                      "live": time.time() - mtime < LIVE_SECONDS and last.get("step") != opts.get("steps"),
                      "has_meta": bool(meta), "latest": evals})

    def list(self):
        rows = [self.summary(n, d) for n, d in self.dirs().items()]
        return sorted(rows, key=lambda r: r["updated"], reverse=True)

    def run(self, name, since=-1):
        d = self.dirs().get(name)
        if d is None:
            return None
        return {"name": name, "path": str(d), "meta": read_json(d / "run.json"),
                "checkpoints": checkpoints(d), "best": (d / "best.pt").is_file(),
                "entries": read_log(d / "log.jsonl", since), "updated": (d / "log.jsonl").stat().st_mtime}


class Handler(SimpleHTTPRequestHandler):
    runs = None

    def __init__(self, *args, **kw):
        super().__init__(*args, directory=str(STATIC), **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, obj, status=200):
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/runs":
            return self.send_json(self.runs.list())
        if url.path.startswith("/api/runs/"):
            q = parse_qs(url.query)
            try:
                since = int(q.get("since", ["-1"])[0])
            except ValueError:
                since = -1
            r = self.runs.run(unquote(url.path[len("/api/runs/"):]), since)
            return self.send_json(r) if r else self.send_json({"error": "no such run"}, 404)
        if url.path.startswith("/api/"):
            return self.send_json({"error": "not found"}, 404)
        return super().do_GET()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler):
        self.address_family = socket.AF_INET6 if ":" in addr[0] else socket.AF_INET
        super().__init__(addr, handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", action="append", help="directory of run directories (repeatable; default /data/primer/runs)")
    ap.add_argument("--host", action="append", help="address to listen on (repeatable; default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    Handler.runs = Runs(a.runs or ["/data/primer/runs"])
    servers = [Server((h, a.port), Handler) for h in a.host or ["127.0.0.1"]]
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"run explorer on {', '.join(f'http://{h}:{a.port}' for h in a.host or ['127.0.0.1'])}", flush=True)
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
