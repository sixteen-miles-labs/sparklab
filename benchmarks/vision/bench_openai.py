#!/usr/bin/env python3
"""Reproducible single-stream image latency and decode benchmark (Pillow required)."""
import argparse
import base64
import hashlib
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

PROMPT = ('Describe the image or images in detail: list the colored shapes, their positions, '
          'and the background. Then explain how their arrangement could be recreated step by step. '
          'Continue with detailed observations until you reach the output limit.')
CASES = {'image_256': [256], 'image_1024': [1024], 'two_images_512': [512, 512]}


def content(sizes):
    parts, hashes = [], []
    for i, size in enumerate(sizes):
        im = Image.new('RGB', (size, size), 'white')
        d = ImageDraw.Draw(im)
        d.rectangle((size//8, size//8, size*3//8, size*3//8), fill='red' if i == 0 else 'blue')
        d.ellipse((size*5//8, size//8, size*7//8, size*3//8), fill='blue' if i == 0 else 'red')
        d.polygon([(size//2, size//2), (size//4, size*7//8), (size*3//4, size*7//8)], fill='green')
        buf = io.BytesIO(); im.save(buf, format='PNG'); raw = buf.getvalue()
        hashes.append(hashlib.sha256(raw).hexdigest())
        parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}})
    return parts + [{'type': 'text', 'text': PROMPT}], hashes


def request(args, parts):
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': parts}],
            'temperature': 0, 'max_tokens': args.tokens, 'ignore_eos': True,
            'chat_template_kwargs': {'enable_thinking': False}, 'stream': True,
            'stream_options': {'include_usage': True}}
    req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.perf_counter(); stamps, pieces, usage, reasons = [], [], None, []
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'): continue
            payload = line[5:].strip()
            if payload == '[DONE]': break
            event = json.loads(payload)
            if event.get('error'): raise RuntimeError(event['error'])
            if event.get('usage'): usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta') or {}
                text = delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning')
                if text: stamps.append(time.perf_counter()); pieces.append(text)
                if choice.get('finish_reason'): reasons.append(choice['finish_reason'])
    elapsed = time.perf_counter() - start
    if not stamps or not usage: raise RuntimeError('Missing streamed content or usage')
    n = usage['completion_tokens']
    if n != args.tokens: raise RuntimeError(f'Expected {args.tokens} output tokens; got {n}')
    return {'ttft_s': stamps[0]-start, 'e2e_s': elapsed,
            'decode_tok_s': (n-1)/(stamps[-1]-stamps[0]), 'output_tok_s': n/elapsed,
            'usage': usage, 'content_chunks': len(stamps), 'finish_reasons': reasons,
            'output': ''.join(pieces)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--url', required=True)
    p.add_argument('--model', default='qwen36-vision-bench'); p.add_argument('--profile', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--trials', type=int, default=3); args = p.parse_args()
    report = {'profile': args.profile, 'method': {'concurrency': 1, 'trials': args.trials,
              'warmups_per_case': 1, 'output_tokens': args.tokens, 'ignore_eos': True,
              'temperature': 0, 'enable_thinking': False, 'prompt': PROMPT,
              'ttft': 'Request start to first nonempty content/reasoning SSE chunk.',
              'decode': '(usage completion tokens - 1) / (last content chunk time - first content chunk time); chunk-based approximation, especially with MTP.',
              'e2e': 'Request start through final streamed usage and DONE.',
              'cache_policy': 'Disable vLLM prefix and multimodal processor caches; SparkLab image prefix reuse is disabled.'}, 'cases': {}}
    for name, sizes in CASES.items():
        parts, hashes = content(sizes)
        warmup = request(args, parts)
        rows = [request(args, parts) for _ in range(args.trials)]
        metrics = {k: statistics.median(r[k] for r in rows) for k in ('ttft_s', 'e2e_s', 'decode_tok_s', 'output_tok_s')}
        report['cases'][name] = {'image_sizes': sizes, 'image_sha256': hashes, 'warmup': warmup, 'trials': rows, 'median': metrics}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(name, json.dumps(metrics), flush=True)


if __name__ == '__main__': main()
