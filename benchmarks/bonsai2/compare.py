"""Reproducible single-client Bonsai text/image smoke screen and latency probes.

Run each backend serially with identical requests. This is a small integration
screen, not a general model-quality evaluation.
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def image_url(text="SPARK 427", color="red"):
    im = Image.new("RGB", (640, 384), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((20, 20, 200, 200), fill=color)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 44)
    d.text((30, 270), text, fill="black", font=font)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def screen(base):
    cases = [
        ("arithmetic", "What is 17 times 23? Answer with just the number.", "391"),
        (
            "capital",
            "What is the capital of France? Answer with just the city.",
            "paris",
        ),
        ("json", 'Return only valid JSON with key "ok" set to true.', '"ok"'),
        ("coding", "Write a Python function square(x) that returns x*x.", "return"),
        (
            "reasoning",
            "If all dax are wugs and no wugs are zibs, can a dax be a zib? Answer yes or no.",
            "no",
        ),
        (
            "ocr",
            [
                {
                    "type": "text",
                    "text": "Read the text printed at the bottom of this image. Reply with only that text.",
                },
                {"type": "image_url", "image_url": {"url": image_url()}},
            ],
            "spark 427",
        ),
        (
            "color",
            [
                {
                    "type": "text",
                    "text": "What color is the square? Reply with one word.",
                },
                {"type": "image_url", "image_url": {"url": image_url()}},
            ],
            "red",
        ),
        (
            "changed_image",
            [
                {
                    "type": "text",
                    "text": "Read the text printed at the bottom of this image. Reply with only that text.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url("BIRCH 918", "blue")},
                },
            ],
            "birch 918",
        ),
    ]
    out = []
    for name, content, answer in cases:
        body = {
            "model": "bonsai2",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        start = time.perf_counter()
        r = requests.post(base + "/v1/chat/completions", json=body, timeout=300)
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        row = {
            "case": name,
            "passed": r.ok and answer in text.lower(),
            "elapsed_seconds": time.perf_counter() - start,
            "response": data,
        }
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = screen(args.url)
    Path(args.output).write_text(
        json.dumps({"url": args.url, "cases": result}, indent=2) + "\n"
    )
