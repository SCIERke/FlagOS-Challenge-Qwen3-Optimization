#!/usr/bin/env python3
"""Minimal upload server for sync bundles.

POST /upload (multipart/form-data, field name: file)
Saves bundle under /tmp and applies it with apply_bundle.sh.
Apply runs asynchronously to avoid HTTP gateway timeout.
"""

from __future__ import annotations

import cgi
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import os
import subprocess
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse


WORKSPACE = Path(__file__).resolve().parents[3]
APPLY_SH = WORKSPACE / "src/tools/sync/apply_bundle.sh"
UPLOAD_DIR = Path("/tmp/flagos_sync")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/tmp/flagos_sync_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    def _status_for_stem(self, stem: str) -> tuple[str, str]:
        log_path = LOG_DIR / f"{stem}.log"
        if not log_path.exists():
            return "unknown", ""
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "[apply] done" in text:
            return "done", str(log_path)
        if "[apply] no valid selection" in text or "ERROR" in text:
            return "failed", str(log_path)
        return "running", str(log_path)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if parsed.path == "/status":
            qs = parse_qs(parsed.query)
            stem = qs.get("bundle", [""])[0]
            if not stem:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(
                    b"missing query param: bundle (e.g. /status?bundle=sync_bundle_YYYYMMDD_HHMMSS_XXXXXX)"
                )
                return
            status, log_path = self._status_for_stem(stem)
            self.send_response(200)
            self.end_headers()
            msg = f"bundle={stem}\nstatus={status}\n"
            if log_path:
                msg += f"log={log_path}\n"
            self.wfile.write(msg.encode("utf-8", errors="replace"))
            return

        if parsed.path == "/status/latest":
            logs = sorted(LOG_DIR.glob("sync_bundle_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not logs:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"no bundle logs yet")
                return
            latest = logs[0]
            stem = latest.stem
            status, log_path = self._status_for_stem(stem)
            self.send_response(200)
            self.end_headers()
            msg = f"bundle={stem}\nstatus={status}\nlog={log_path}\n"
            self.wfile.write(msg.encode("utf-8", errors="replace"))
            return

        if parsed.path == "/status/all":
            logs = sorted(
                LOG_DIR.glob("sync_bundle_*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            self.send_response(200)
            self.end_headers()
            if not logs:
                self.wfile.write(b"no bundle logs yet\n")
                return
            rows = []
            for p in logs:
                stem = p.stem
                status, log_path = self._status_for_stem(stem)
                rows.append(f"bundle={stem}\tstatus={status}\tlog={log_path}")
            self.wfile.write(("\n".join(rows) + "\n").encode("utf-8", errors="replace"))
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/upload":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST"},
        )
        if "file" not in form:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"missing field: file")
            return

        file_item = form["file"]
        if not file_item.file:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"invalid file")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = UPLOAD_DIR / f"sync_bundle_{stamp}_{time.time_ns() % 1000000:06d}.tgz"
        with dst.open("wb") as f:
            f.write(file_item.file.read())

        log_path = LOG_DIR / f"{dst.stem}.log"
        log_fp = log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env["FORCE_NON_INTERACTIVE"] = "1"
        subprocess.Popen(
            ["bash", str(APPLY_SH), str(dst)],
            cwd=str(WORKSPACE),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self.send_response(202)
        self.end_headers()
        status_url = f"/status?bundle={dst.stem}"
        msg = (
            f"accepted\nbundle={dst}\n"
            f"log={log_path}\n"
            f"status_url={status_url}\n"
            "note=apply is running asynchronously\n"
        )
        self.wfile.write(msg.encode("utf-8", errors="replace"))


def main() -> None:
    server = HTTPServer(("0.0.0.0", 30000), Handler)
    print("sync receiver listening on :30000, endpoint POST /upload", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
