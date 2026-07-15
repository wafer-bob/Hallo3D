"""Local LMM service for Hallo3D Multi-modal Hallucination Detection.

Loads a LLaVA-v1.6 vision-language model (default:
llava-hf/llava-v1.6-34b-hf, as deployed in the paper; smaller variants
such as llava-v1.6-vicuna-13b-hf / -7b-hf work as drop-in replacements
via --model-id) and exposes a minimal HTTP endpoint:

    POST /query   {"image_b64": <base64 PNG>, "prompt": <inquiry P_I>}
    -> {"response": <raw LMM answer>}

Run (choose a free GPU):
    CUDA_VISIBLE_DEVICES=0 python serve_lmm.py --port 39121

Only stdlib + transformers are required (no fastapi/flask dependency).
"""

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image


def load_model(model_id: str, dtype: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="cuda:0"
    ).eval()
    return processor, model


class LMMWorker:
    """Serialized (single GPU) inference wrapper."""

    def __init__(self, model_id: str, dtype: str, max_new_tokens: int):
        self.processor, self.model = load_model(model_id, dtype)
        self.max_new_tokens = max_new_tokens
        self.lock = threading.Lock()

    @torch.inference_mode()
    def answer(self, image: Image.Image, prompt: str, history=None) -> str:
        messages = []
        for i, (user_text, assistant_text) in enumerate(history or []):
            user_content = [{"type": "text", "text": user_text}]
            if i == 0:
                user_content.insert(0, {"type": "image"})
            messages.append({"role": "user", "content": user_content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )
        user_content = [{"type": "text", "text": prompt}]
        if not history:
            user_content.insert(0, {"type": "image"})
        messages.append({"role": "user", "content": user_content})
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.model.device)
        with self.lock:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


def make_handler(worker: LMMWorker):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/query":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                image = Image.open(
                    io.BytesIO(base64.b64decode(req["image_b64"]))
                ).convert("RGB")
                response = worker.answer(
                    image, req["prompt"], req.get("history")
                )
                self._send(200, {"response": response})
            except Exception as e:
                self._send(500, {"error": repr(e)})

        def log_message(self, fmt, *args):
            print(f"[serve_lmm] {self.address_string()} {fmt % args}")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", default="llava-hf/llava-v1.6-34b-hf"
    )
    parser.add_argument("--port", type=int, default=39121)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"[serve_lmm] loading {args.model_id} ...")
    worker = LMMWorker(args.model_id, args.dtype, args.max_new_tokens)
    print(f"[serve_lmm] ready on http://{args.host}:{args.port}")
    ThreadingHTTPServer(
        (args.host, args.port), make_handler(worker)
    ).serve_forever()


if __name__ == "__main__":
    main()
