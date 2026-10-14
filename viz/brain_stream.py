"""Live brain viewer feed: serves viz/brain_viewer.html and streams every
100 ms brain slice to it (Server-Sent Events), and/or records the slices to
a .jsonl.gz file that viz/replay.py can play back later.

Used by harness.py (--brain-viz, --record-brain). Standard library only;
the server runs in a background thread and never blocks the fight loop
(slow or absent viewers just miss frames).

One frame (JSON):
  t        game seconds since the boss spawned
  spk      base64 uint32 indices of every neuron that spiked in the slice
  cnt      base64 uint8 spike counts for those neurons (same order)
  senses   {"LC11_right": Hz, ...}  the visual populations the senses drove
  drive    [sideways, vertical]     the readout's output (+ = right / climb)
  keys     keys held this tick, who pressed them ("brain", "assist", ...)
  hp, max_hp, boss  (boss = fraction of Moon Lord HP left), phase
"""
import base64
import gzip
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIEWER = HERE / "brain_viewer.html"


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def encode_frame(rates, t, senses=None, drive=None, keys=None, state=None, phase="fight", slice_ms=100.0):
    """rates: every neuron's rate (Hz) over the slice (the engine's stimulate()
    output), or None when the brain wasn't stepped (a scripted Nurse trip)."""
    fr = {"t": round(float(t), 2), "phase": phase}
    if rates is not None:
        rates = np.asarray(rates)
        idx = np.flatnonzero(rates > 0)
        cnt = np.clip(np.rint(rates[idx] * slice_ms / 1000.0), 1, 255).astype(np.uint8)
        fr["spk"], fr["cnt"] = _b64(idx.astype(np.uint32)), _b64(cnt)
        fr["n_active"] = int(len(idx))
    if senses:
        fr["senses"] = {f"{k[0]}_{'left' if k[1] < 0 else 'right'}": round(float(v), 1)
                        for k, v in senses.items() if v > 0}
    if drive is not None:
        fr["drive"] = [round(float(drive[0]), 3), round(float(drive[1]), 3)]
    if keys is not None:
        fr["keys"] = keys
    if state:
        p = state.get("player") or {}
        fr["hp"], fr["max_hp"] = p.get("hp"), p.get("max_hp")
        parts = [n for n in state.get("npcs", []) if n.get("moon_lord_part")]
        if parts:
            fr["boss"] = round(sum(n.get("hp", 0) for n in parts) / max(1, sum(n.get("max_hp", 1) for n in parts)), 3)
    return fr


class _Hub:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.last = None   # last frame, so a viewer that opens late isn't blank

    def add(self):
        q = queue.Queue(maxsize=50)
        with self.lock:
            self.clients.append(q)
        return q

    def remove(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def publish(self, msg):
        self.last = msg
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass  # viewer too slow: drop the frame


class BrainServer:
    """http://localhost:<port>/  the viewer; /layout  neuron positions; /stream  frames."""

    def __init__(self, layout_json, port=8765, host="127.0.0.1"):
        self.hub = _Hub()
        layout_gz = gzip.compress(layout_json.encode(), 5)
        hub = self.hub

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    body = VIEWER.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/layout":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Encoding", "gzip")
                    self.send_header("Content-Length", str(len(layout_gz)))
                    self.end_headers()
                    self.wfile.write(layout_gz)
                elif path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    q = hub.add()
                    try:
                        if hub.last:
                            self.wfile.write(f"data: {hub.last}\n\n".encode())
                        while True:
                            try:
                                msg = q.get(timeout=5)
                                self.wfile.write(f"data: {msg}\n\n".encode())
                            except queue.Empty:
                                self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        hub.remove(q)
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://localhost:{port}/"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def publish(self, frame):
        self.hub.publish(json.dumps(frame, separators=(",", ":")))

    def close(self):
        self.httpd.shutdown()


class BrainObserver:
    """harness.run_episode(observer=...) hook: turns each tick into a frame for
    the live viewer and/or the recording."""

    def __init__(self, server=None, record_path=None, meta=None):
        self.server = server
        self.rec = gzip.open(record_path, "wt") if record_path else None
        self.record_path = record_path
        if self.rec:
            self.rec.write(json.dumps({"header": True, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       **(meta or {})}) + "\n")
        self.t0 = None

    def _emit(self, frame):
        if self.server:
            self.server.publish(frame)
        if self.rec:
            frame = {**frame, "wall": round(time.monotonic() - self.t0, 3)}
            self.rec.write(json.dumps(frame, separators=(",", ":")) + "\n")

    def tick(self, policy, state, elapsed, action=None, sources=None, brain_stepped=True, phase="fight"):
        if self.t0 is None:
            self.t0 = time.monotonic()
        keys = {k: (sources or {}).get(k, "?") for k, v in (action or {}).items() if v}
        rates = getattr(policy, "last_spikes", None) if brain_stepped else None
        drive = getattr(policy, "last_drive", None) if brain_stepped else None
        senses = getattr(policy, "last_senses", None) if brain_stepped else None
        self._emit(encode_frame(rates, elapsed, senses, drive, keys, state, phase))

    def end(self, outcome, row=None):
        frame = {"t": row.get("seconds") if row else None, "phase": "end", "outcome": outcome}
        if row:
            frame.update(hits=row.get("hits"), damage=row.get("damage_taken"), nurse_trips=row.get("nurse_trips"))
        self._emit(frame) if self.t0 is not None else None

    def close(self):
        if self.rec:
            self.rec.close()
            self.rec = None
