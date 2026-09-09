#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Expanded Qwen quality screen, reusing the shared serving regression client.

HumanEval programs run in a CPU-only, network-isolated, read-only container with
resource limits. Full prompts and responses belong in the supplied results path.
"""
import importlib.util
import json
import os
import re
import subprocess
import uuid
from decimal import Decimal
from pathlib import Path

# The recorded run used the locally built Qwen3.6 image's Python interpreter.
# Reproduction on another host can supply a pinned image containing python3.
IMAGE = os.environ.get(
    'SPARKLAB_QUALITY_IMAGE',
    'sha256:a563337b36f29e11803737bca3c8833d6ed8d66a2efd244348cc9dc1bd53ba5e',
)
RUNNER = '''import contextlib,io,json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(4,4))
resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
payload=json.load(sys.stdin)
namespace={}
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
 exec(compile(payload['program']+'\\n'+payload['test']+'\\ncheck('+payload['entry_point']+')','<quality-case>','exec'),namespace)
print('CHECK_PASSED')
'''


def evaluate_program(text, case):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S | re.I)
    program = (blocks[-1] if blocks else text).strip()
    name = 'qwen27-eval-' + uuid.uuid4().hex[:12]
    command = [
        'docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '32',
        '--memory', '256m', '--memory-swap', '256m', '--cpus', '1', '--user', '65534:65534',
        '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
        '-i', '--entrypoint', 'python3', IMAGE, '-c', RUNNER,
    ]
    try:
        result = subprocess.run(command, input=json.dumps({
            'program': program, 'test': case['test'], 'entry_point': case['entry_point'],
        }), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=12)
        return {'passed': result.returncode == 0 and result.stdout.strip() == 'CHECK_PASSED',
                'returncode': result.returncode, 'output': result.stdout[-4096:], 'image': IMAGE}
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {'passed': False, 'error': 'execution timed out', 'image': IMAGE}


def main():
    source = Path(__file__).resolve().parents[1] / 'qwen36_marlin' / 'validate_serving.py'
    spec = importlib.util.spec_from_file_location('serving_validation', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_score = module.score

    def score(case, result):
        if case['kind'] not in ('gsm8k', 'humaneval'):
            return original_score(case, result)
        choice = result['response']['choices'][0]
        if choice.get('finish_reason') == 'length':
            return False
        text = (choice['message'].get('content') or '').strip()
        if case['kind'] == 'gsm8k':
            answers = re.findall(r'####\s*([-+]?\d[\d,.]*)', text)
            return bool(answers) and Decimal(answers[-1].replace(',', '').rstrip('.')) == Decimal(case['expected'])
        execution = evaluate_program(text, case)
        result['execution'] = execution
        return execution['passed']

    module.score = score
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())
