#!/usr/bin/env python3
"""Fixed-input Qwen3.6 quality and serving checks against an existing server.

The corpus is generated once by prepare_validation.py and shared across profiles.
This is a regression/capability gate, not a general model-quality benchmark.
No model-generated program is executed outside the bounded AST evaluator below.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, body, cancel=False, cancel_before_tokens=False):
    start = time.monotonic()
    req = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1200) as response:
        if cancel_before_tokens:
            time.sleep(0.05)
            return {'disconnected_before_reading_tokens':True, 'seconds':time.monotonic()-start}
        if not body.get('stream'):
            result = json.load(response)
        else:
            content, reasoning, events = [], [], 0
            first = last = None
            usage, finish = {}, None
            done = False
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    text = delta.get('content') or ''
                    thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if text or thought:
                        last = time.monotonic()
                        first = first or last
                        events += 1
                    content.append(text)
                    reasoning.append(thought)
                    finish = choice.get('finish_reason') or finish
                if cancel and events >= 2:
                    break
            result = {'choices': [{'message': {'role': 'assistant',
                'content': ''.join(content), 'reasoning_content': ''.join(reasoning)},
                'finish_reason': finish}], 'usage': usage, 'stream_done': done,
                'cancelled_after_events': events if cancel else None,
                'ttft_seconds': first - start if first else None,
                'decode_tok_s': (usage.get('completion_tokens', 0) - 1) / (last-first)
                    if first and last > first and usage else None}
    return {'response': result, 'seconds': time.monotonic() - start}


def evaluate_code(text, case):
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S | re.I)
    tree = ast.parse((blocks[-1] if blocks else text).strip())
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.If,
        ast.Return, ast.Compare, ast.Name, ast.Load, ast.Constant, ast.Lt,
        ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.BinOp, ast.Add,
        ast.Sub, ast.Mult, ast.Mod, ast.FloorDiv, ast.UnaryOp, ast.USub,
        ast.BoolOp, ast.And, ast.Or, ast.Not, ast.IfExp, ast.Expr)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    fn = tree.body[0]
    if fn.name != case['function'] or [a.arg for a in fn.args.args] != case['args']:
        return False
    if any(not isinstance(n, allowed) for n in ast.walk(tree)):
        return False
    # No calls, loops, attributes, subscripts, imports or recursion are allowed.
    namespace = {'__builtins__': {}}
    exec(compile(tree, '<bounded-model-function>', 'exec'), namespace)
    return all(namespace[fn.name](*args) == expected for args, expected in case['tests'])


def typed_equal(actual, expected):
    """JSON equality must not accept True as 1 or a float as an integer."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(typed_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(typed_equal(x, y) for x, y in zip(actual, expected))
    return actual == expected


def score(case, result):
    r = result['response']
    msg = r['choices'][0]['message']
    text = (msg.get('content') or '').strip()
    if r['choices'][0].get('finish_reason') == 'length':
        return False
    kind = case['kind']
    if kind == 'json':
        return typed_equal(json.loads(text), case['expected'])
    if kind == 'code':
        return evaluate_code(text, case)
    if kind == 'tool':
        calls = msg.get('tool_calls') or []
        return len(calls) == 1 and calls[0]['function']['name'] == 'multiply' and typed_equal(json.loads(calls[0]['function']['arguments']), case['expected'])
    if kind in ('reasoning', 'aime'):
        answers = re.findall(r'\\boxed\{([^{}]+)\}', text)
        return bool(msg.get('reasoning_content') or msg.get('reasoning')) and bool(answers) and answers[-1].strip() == str(case['expected'])
    return text == str(case['expected'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:1929')
    p.add_argument('--model', default='qwen36-bench')
    p.add_argument('--server-kind', choices=('sparklab','vllm'), default='sparklab')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration-seconds', type=float, default=0)
    p.add_argument('--cgroup', type=Path)
    p.add_argument('--quality-only', action='store_true')
    p.add_argument('--skip-quality', action='store_true', help='Run operational checks and soak after a separately recorded quality screen')
    a = p.parse_args()
    if a.quality_only and a.skip_quality:
        p.error('--quality-only and --skip-quality are mutually exclusive')
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = json.loads(a.corpus.read_text())
    stop = threading.Event()
    def get(path):
        with urllib.request.urlopen(a.base_url + path, timeout=20) as r:
            if path == '/health' and a.server_kind == 'vllm':
                return {'status':'ok', 'source':'HTTP 200'}
            return json.load(r)
    def write(name, row):
        with (a.output / (name + '.jsonl')).open('a') as f:
            f.write(json.dumps(row) + '\n')
    def monitor():
        while not stop.is_set():
            row = {'time': time.time()}
            try:
                row['memory'] = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if len(s.split()) == 3}
                row['swap_io'] = {s.split()[0]: int(s.split()[1]) for s in Path('/proc/vmstat').read_text().splitlines() if s.startswith(('pswpin ', 'pswpout '))}
                if a.cgroup:
                    row['cgroup'] = {name: (a.cgroup/name).read_text().strip() for name in ('memory.current', 'memory.swap.current', 'memory.events', 'io.stat')}
                row['health'] = get('/health')
                if a.server_kind == 'sparklab':
                    row['stats'] = get('/v1/stats')
            except Exception as exc:
                row['error'] = repr(exc)
            write('telemetry', row)
            stop.wait(5)
    initial_health = get('/health')
    start = time.monotonic()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    results, errors = [], []
    def run(case, stage='quality'):
        body = dict(case['body'], model=a.model)
        row = {'id': case['id'], 'kind': case['kind'], 'payload': body, 'started': time.time()}
        try:
            row.update(request(a.base_url, body))
            row['passed'] = score(case, row)
        except Exception as exc:
            row.update(passed=False, error=repr(exc))
        write(stage, row)
        results.append({'id': row['id'], 'kind': row['kind'], 'passed': row['passed']})
        print(stage, row['id'], row['passed'], flush=True)
        return row
    try:
        for case in ([] if a.skip_quality else corpus['cases']):
            row = run(case)
            if case['kind'] == 'tool' and row['passed']:
                msg = row['response']['choices'][0]['message']
                call = msg['tool_calls'][0]
                args = json.loads(call['function']['arguments'])
                follow = {'id': case['id']+'-roundtrip', 'kind': 'exact', 'expected': str(args['a']*args['b']),
                    'body': dict(case['body'], messages=case['body']['messages']+[msg, {'role':'tool', 'tool_call_id':call['id'], 'content':str(args['a']*args['b'])}, {'role':'user','content':'Reply with only the integer result.'}], chat_template_kwargs={'enable_thinking':False})}
                run(follow)
        if not a.quality_only:
            for case in corpus['contexts']:
                row = run(case, 'context')
                exact = row.get('response', {}).get('usage', {}).get('prompt_tokens') == case['tokens']
                results[-1]['exact_prompt_tokens'] = exact
                results[-1]['passed'] &= exact
            for limit in (1,2,3,4,5,7,16,65,256):
                body = {'model':a.model, 'messages':[{'role':'user','content':'Count upward from one, separated by commas.'}], 'temperature':0, 'max_completion_tokens':limit, 'ignore_eos':True, 'stream':True, 'stream_options':{'include_usage':True}, 'chat_template_kwargs':{'enable_thinking':False}}
                row = request(a.base_url, body)
                ok = row['response']['usage']['completion_tokens'] == limit and row['response']['stream_done'] and row['response']['choices'][0]['finish_reason'] == 'length'
                write('budgets', dict(limit=limit, payload=body, passed=ok, **row))
                results.append({'id':f'budget-{limit}', 'kind':'runtime', 'passed':ok})
            cases = [c for c in corpus['cases'] if c['kind'] == 'exact'][:4]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda c: run(c, 'queued'), cases))
            for i in range(3):
                body = {'model':a.model, 'messages':[{'role':'user','content':f'Count upward from {i}, continuing for as long as possible.'}], 'temperature':0, 'max_completion_tokens':8192, 'ignore_eos':True, 'stream':True, 'chat_template_kwargs':{'enable_thinking':False}}
                row = request(a.base_url, body, cancel=True)
                write('cancellation', dict(payload=body, **row))
                results.append({'id':f'cancel-{i}', 'kind':'runtime', 'passed':row['response']['cancelled_after_events'] >= 2})
                recovered = run(cases[i], 'after-cancellation')
                results[-1]['passed'] &= recovered.get('seconds', 1000) < 30
            long_case = corpus['contexts'][-1]
            body = dict(long_case['body'], model=a.model, stream=True)
            row = request(a.base_url, body, cancel_before_tokens=True)
            write('early-disconnect', dict(payload=body, **row))
            recovered = run(cases[0], 'after-early-disconnect')
            results[-1]['passed'] &= recovered.get('seconds', 1000) < 30
            for temperature in (0.7, 1.0):
                body = dict(cases[0]['body'], model=a.model, temperature=temperature,
                    ignore_eos=True, max_completion_tokens=32, stream=True,
                    stream_options={'include_usage':True})
                row = request(a.base_url, body)
                ok = row['response']['stream_done'] and row['response']['usage']['completion_tokens'] == 32
                write('sampled-fallback', dict(payload=body, passed=ok, **row))
                results.append({'id':f'sampled-{temperature}', 'kind':'runtime', 'passed':ok})
            body = dict(long_case['body'], model=a.model)
            body['messages'] = [dict(body['messages'][0])]
            body['messages'][0]['content'] += ' x' * 256
            try:
                row = request(a.base_url, body)
                row['passed'] = False
            except urllib.error.HTTPError as exc:
                error = json.load(exc)
                row = {'status':exc.code, 'error_response':error,
                    'passed':exc.code == 400 and error.get('error', {}).get('code') == 'context_length_exceeded'}
            write('overlength', dict(payload=body, **row))
            results.append({'id':'overlength-rejected', 'kind':'runtime', 'passed':row['passed']})
            run(cases[1], 'after-overlength')
            i = 0
            while time.monotonic() - start < a.duration_seconds:
                soak_contexts = [c for c in corpus['contexts'] if c['tokens'] <= 8192]
                case = soak_contexts[(i // 10) % len(soak_contexts)] if i % 10 == 0 else cases[i % len(cases)]
                # Distinct suffixes exercise cache eviction and request ownership.
                body = dict(case['body'], model=a.model, max_completion_tokens=256,
                    ignore_eos=True, stream=True, stream_options={'include_usage':True})
                body['messages'] = [dict(case['body']['messages'][0])]
                body['messages'][0]['content'] += f'\nRun marker: {i}. Explain briefly before answering.'
                row = request(a.base_url, body)
                ok = row['response']['usage']['completion_tokens'] == 256 and row['response']['stream_done']
                write('soak', dict(index=i, payload=body, passed=ok, **row))
                if not ok:
                    raise RuntimeError(f'soak request {i} failed protocol checks')
                i += 1
                if i % 20 == 0:
                    print('soak', i, 'elapsed', round(time.monotonic()-start), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(repr(exc))
    finally:
        stop.set()
        thread.join(25)
        try:
            final_health = get('/health')
        except Exception as exc:
            final_health = {'error':repr(exc)}
            errors.append(repr(exc))
        summary = {'corpus_sha256':hashlib.sha256(a.corpus.read_bytes()).hexdigest(),
            'duration_seconds':time.monotonic()-start, 'requested_duration_seconds':a.duration_seconds,
            'initial_health':initial_health, 'final_health':final_health,
            'results':results, 'errors':errors, 'passed':not errors and all(r['passed'] for r in results)}
        (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
