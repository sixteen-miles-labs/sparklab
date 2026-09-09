#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the frozen Qwen27 quality screen from hash-checked local datasets."""
import argparse
import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path

AIME_SHA256 = 'b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331'
SEED = 20260909


def context(tokenizer, target, position, output=128):
    def count(messages):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        return len(encoded['input_ids'] if hasattr(encoded, 'keys') else encoded)

    secret = f'SPRUCE-{target}-492618'
    needle = f'The archive access code is {secret}.\n'
    line = 'An ordinary archive entry describes the river, valley, village, and its seasonal weather.\n'

    def messages(n, padding=0):
        left = {'start': 0, 'middle': n // 2, 'end': n}[position]
        return [{'role': 'user', 'content': 'Read the archive and find its access code.\n'
                 + line * left + needle + line * (n - left) + ' x' * padding
                 + '\nReply with only the archive access code.'}]

    lo, hi, best = 0, target // 10, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(messages(mid)) <= target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    # Keep padding nonempty to avoid a tokenizer boundary change at zero padding.
    best = max(0, best - 2)
    padding = target - count(messages(best))
    for _ in range(8):
        msgs = messages(best, padding)
        delta = target - count(msgs)
        if not delta:
            break
        padding += delta
    if count(msgs) != target:
        raise ValueError(f'cannot construct exactly {target} tokens')
    return {'id': f'context-{target}-{position}', 'kind': 'exact', 'expected': secret,
            'tokens': target, 'prompt_tokens': target, 'body': {
                'messages': msgs, 'temperature': 0, 'max_completion_tokens': output,
                'chat_template_kwargs': {'enable_thinking': False}}}


def checked_rows(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'dataset SHA256 mismatch: {path}')
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def build(tokenizer, aime, gsm8k, humaneval):
    folder = Path(__file__).resolve().parent
    sources = json.loads((folder / 'quality_sources.json').read_text())
    checked_rows(aime, AIME_SHA256)
    source = folder.parent / 'qwen36_marlin' / 'prepare_validation.py'
    spec = importlib.util.spec_from_file_location('shared_corpus', source)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    inherited = shared.build(tokenizer, aime)
    cases = inherited['cases'][:20]
    for name in ('arithmetic-1', 'reasoning-0', 'json-0', 'code-clamp'):
        case = copy.deepcopy(next(c for c in cases if c['id'] == name))
        case['id'] = 'long-' + name
        case['body']['messages'][0]['content'] = (
            'Background notes for a formatting exercise. Ignore these repeated notes when answering the final task.\n'
            + 'The river crosses the valley beside a quiet village.\n' * 700
            + '\nFINAL TASK:\n' + case['body']['messages'][0]['content']
        )
        cases.append(case)
    rng = random.Random(SEED)
    math = checked_rows(gsm8k, sources['gsm8k']['jsonl_sha256'])
    for i in sorted(rng.sample(range(len(math)), 20)):
        row = math[i]
        cases.append({'id': f'gsm8k-{i}', 'kind': 'gsm8k',
                      'expected': row['answer'].split('####')[-1].strip().replace(',', ''),
                      'body': {'messages': [{'role': 'user', 'content': row['question']
                               + '\nSolve the problem. End your final answer with #### followed by the numerical result.'}],
                               'temperature': 0, 'max_completion_tokens': 2048,
                               'chat_template_kwargs': {'enable_thinking': True}}})
    code = checked_rows(humaneval, sources['humaneval']['jsonl_sha256'])
    for i in sorted(rng.sample(range(len(code)), 20)):
        row = code[i]
        cases.append({'id': row['task_id'], 'kind': 'humaneval',
                      'entry_point': row['entry_point'], 'test': row['test'],
                      'body': {'messages': [{'role': 'user', 'content':
                               'Complete the following Python programming task. Return a complete standalone module in exactly one Python code block, including necessary imports. Do not include tests or explanations.\n\n'
                               + row['prompt']}], 'temperature': 0, 'max_completion_tokens': 2048,
                               'chat_template_kwargs': {'enable_thinking': False}}})
    for case in inherited['cases'][20:]:
        case['body']['max_completion_tokens'] = 8192
        cases.append(case)
    cases += [context(tokenizer, n, position) for n, position in (
        (16384, 'start'), (32768, 'middle'), (64512, 'end'),
    )]
    return {
        'description': 'Fixed regression and long-prompt cases; 20 seeded GSM8K test problems; 20 seeded HumanEval tasks; first five AIME-25 problems at 8192-token budget; 16K/32K/~64K recall. No quality-equivalence claim.',
        'sources': sources, 'seed': SEED, 'cases': cases, 'contexts': [],
        'context_builder_fix': 'Count input_ids in Transformers BatchEncoding; the three oversized context cases in v1 are invalid and replaced.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True)
    for name in ('aime', 'gsm8k', 'humaneval', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--operational-output', type=Path)
    args = parser.parse_args()
    for path in (args.output, args.operational_output):
        if path is not None and path.exists():
            raise FileExistsError(path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    corpus = build(tokenizer, args.aime, args.gsm8k, args.humaneval)
    args.output.write_text(json.dumps(corpus, indent=2) + '\n')
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
    if args.operational_output is not None:
        operational = {'cases': corpus['cases'][:24], 'contexts': [
            context(tokenizer, n, 'middle', 32) for n in (1024, 8192, 16384, 32768, 65504)
        ]}
        args.operational_output.write_text(json.dumps(operational, indent=2) + '\n')
        print(hashlib.sha256(args.operational_output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
