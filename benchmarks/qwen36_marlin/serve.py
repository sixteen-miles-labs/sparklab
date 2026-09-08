#!/usr/bin/env python3
"""Launch the fixed Qwen3.6 GB10 container profile with a separate Marlin FTW."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex


def command(model: Path, image: str, port: int) -> list[str]:
    model = model.resolve(strict=True)
    index = json.loads((model / 'freetoken_weight.json').read_text())
    if index.get('quant_format') != 'nvfp4_marlin' or hashlib.sha256((model / 'config.json').read_bytes()).hexdigest() != '58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc':
        raise ValueError('This profile requires the separate pinned Qwen3.6 Marlin FTW artifact; see README.md for conversion.')
    for shard in index['shards']:
        path = (model / shard['file']).resolve(strict=True)
        if path.parent != model or path.stat().st_size != shard['nbytes']:
            raise ValueError(f'Invalid or incomplete FTW shard: {shard["file"]}')
    return ['docker', 'run', '--rm', '--gpus', 'all', '--network', 'host',
        '--shm-size', '16g', '--memory', '96g', '--memory-swap', '96g',
        '--env-file', str(Path(__file__).with_name('gb10.env').resolve()),
        '-v', f'{model}:/artifact:ro', image, 'serve', '--model', '/artifact',
        '--served-model-name', 'qwen36-bench', '--moe-backend', 'offload',
        '--moe-storage', 'disk', '--moe-preload-all', '--moe-cache-rate', '1.0',
        '--nvfp4-backend', 'marlin', '--num-tokens', '32832',
        '--max-seq-len-override', '32832', '--attention-backend', 'fi',
        '--speculative-method', 'mtp', '--speculative-tokens', '4', '--port', str(port)]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='Converted Marlin FTW directory')
    p.add_argument('--image', default='sparklab-qwen36:gb10-v1')
    p.add_argument('--port', type=int, default=1929)
    p.add_argument('--dry-run', action='store_true', help='Check artifact metadata and print the command')
    a = p.parse_args()
    if not 1 <= a.port <= 65535:
        p.error('--port must be between 1 and 65535')
    try:
        cmd = command(a.model, a.image, a.port)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    print(shlex.join(cmd), flush=True)
    if not a.dry_run:
        os.execvp(cmd[0], cmd)
