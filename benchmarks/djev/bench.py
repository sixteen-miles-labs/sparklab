"""Cache-busted three-decision concurrency sweep and text/image smoke checks."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request
import uuid

MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is there an active service outage requiring immediate help?"},
    "route": {"type": "choice", "instructions": "Which team should handle the request?",
              "criteria": {"support": "Outages, errors or technical problems", "billing": "Invoices or payments"}},
    "tone": {"type": "score", "instructions": "How upset is the writer?",
             "criteria": ["calm and polite", "mildly annoyed", "furious"]},
}
CASES = [
    ("All services are down right now! This is outrageous. FIX IT IMMEDIATELY!", True, "support", 2),
    ("Hello, could you please email me last month's invoice? Thank you.", False, "billing", 0),
    ("The whole service is offline. Would you please investigate now? Thank you.", True, "support", 0),
    ("YOU CHARGED ME THREE TIMES! I AM FURIOUS! REFUND ME NOW!", False, "billing", 2),
    ("Hello, how can I update the credit card used for payments? Thanks!", False, "billing", 0),
    ("Nobody can log in and the service is down. Please help immediately, thanks.", True, "support", 0),
    ("The invoice amount is wrong again. This is a little frustrating.", False, "billing", 1),
    ("The service is offline again. This is annoying; please restore it now.", True, "support", 1),
]


def post(url, body):
    start = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        return {"seconds": time.perf_counter() - start, "response": result}
    except (urllib.error.URLError, OSError) as exc:
        detail = exc.read().decode() if isinstance(exc, urllib.error.HTTPError) else str(exc)
        return {"seconds": time.perf_counter() - start, "error": detail}


def body_for(i, run_id):
    return {"model": MODEL, "state": {"ticket": CASES[i % len(CASES)][0], "request_id": f"{run_id}-{i}"},
            "questions": QUESTIONS, "samples": 1, "seed": 42 + i}


def correctness(row, i):
    if "error" in row:
        return [False] * 3
    a = row["response"]["answers"]
    _, urgent, route, tone = CASES[i % len(CASES)]
    return [(a["urgent"]["noul"] >= 0.5) == urgent,
            a["route"]["choice"] == route, round(a["tone"]["score"]) == tone]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    args = parser.parse_args()
    url = args.url.rstrip("/") + "/v1/systemone"
    report = {"model": MODEL, "protocol": "3 decisions; samples=1; cache-busted state; reusable schema; 2 warmup batches per sweep", "sweeps": []}
    for concurrency in args.concurrency:
        run_id = str(uuid.uuid4())
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            warmups = list(pool.map(lambda i: post(url, body_for(i, run_id + '-warmup')), range(2 * concurrency)))
        for warmup in warmups:
            if "error" in warmup:
                raise RuntimeError(warmup["error"])
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(lambda i: post(url, body_for(i, run_id)), range(args.requests)))
        elapsed = time.perf_counter() - start
        times = sorted(r["seconds"] for r in rows)
        checks = [correctness(r, i) for i, r in enumerate(rows)]
        summary = {"concurrency": concurrency, "requests": args.requests, "elapsed_seconds": elapsed,
                   "requests_per_second": sum("error" not in r for r in rows) / elapsed,
                   "decisions_per_second": 3 * sum("error" not in r for r in rows) / elapsed,
                   "latency_mean_seconds": statistics.mean(times), "latency_p50_seconds": statistics.median(times),
                   "latency_p95_seconds": times[min(len(times)-1, int(.95 * len(times)))],
                   "errors": sum('error' in r for r in rows), "correct_per_question": [sum(c[j] for c in checks) for j in range(3)]}
        report["sweeps"].append(summary)
        print(summary, flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    def isolation(i):
        qid = f"flag_{i}"
        body = {"model": MODEL, "state": {"flag": bool(i % 2)}, "samples": 1,
                "questions": {qid: {"type": "noul", "instructions": "Is the flag in the state true?"}}}
        row = post(url, body)
        a = row.get("response", {}).get("answers", {})
        row["correct"] = set(a) == {qid} and (a[qid]["noul"] >= .5) == bool(i % 2)
        return row
    with ThreadPoolExecutor(max_workers=16) as pool:
        isolation_rows = list(pool.map(isolation, range(32)))
    report["isolation"] = {"requests": len(isolation_rows),
                           "correct": sum(row["correct"] for row in isolation_rows),
                           "errors": sum("error" in row for row in isolation_rows)}
    def extra(case, row):
        result = {"case": case, "latency_seconds": row["seconds"]}
        if "error" in row:
            result["error"] = row["error"]
        else:
            result["answers"] = row["response"]["answers"]
        return result
    extras = []
    for samples in (4, 'auto'):
        body = body_for(0, str(uuid.uuid4()))
        body['samples'] = samples
        extras.append(extra(f"samples-{samples}", post(url, body)))
    body = body_for(1, str(uuid.uuid4()))
    body['steps'] = 2
    extras.append(extra("two-denoise-steps", post(url, body)))
    from PIL import Image
    for color in ('red', 'blue'):
        stream = io.BytesIO()
        Image.new('RGB', (128,128), color).save(stream, format='PNG')
        body = {"model": MODEL, "state": "Inspect the image.", "samples": 1,
                "images": ['data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()],
                "questions": {"color": {"type": "choice", "instructions": "What is the color of the image?",
                                          "criteria": {"red": "red", "blue": "blue"}}}}
        extras.append(extra(f"image-{color}", post(url, body)))
    report['extras'] = extras
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
