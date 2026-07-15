"""Mock LMM server for smoke-testing the Hallo3D pipeline without LLaVA.

Always reports a fixed hallucination so that P_E^- extraction and L_CG are
exercised end-to-end.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RESPONSE = (
    "Negative Prompt: 'multi-head, duplicated features, incongruous "
    "perspective, distorted body, extra limbs'"
)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"status": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._send(200, {"response": RESPONSE})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 39121
    print(f"[mock_lmm] serving on :{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
