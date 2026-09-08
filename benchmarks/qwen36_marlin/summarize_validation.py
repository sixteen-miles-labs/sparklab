#!/usr/bin/env python3
"""Rescore both profiles with one grader and audit a separate endurance run."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from validate_serving import score


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def quality(path, corpus):
    cases = {c['id']:c for c in corpus['cases']}
    result, counts = {}, collections.defaultdict(lambda: {'correct':0,'total':0,'truncated':0})
    for row in rows(path/'quality.jsonl'):
        case = cases.get(row['id'])
        if case is None:
            # Roundtrip cases are derived from the same fixed multiply inputs.
            original = cases[row['id'].removesuffix('-roundtrip')]
            args = original['expected']
            case = {'kind':'exact','expected':str(args['a']*args['b'])}
        try:
            passed = score(case, row)
        except Exception:
            passed = False
        response = row.get('response', {})
        finish = (response.get('choices') or [{}])[0].get('finish_reason')
        bucket = counts[row['kind']]
        bucket['correct'] += int(passed)
        bucket['total'] += 1
        bucket['truncated'] += int(finish == 'length')
        result[row['id']] = {'passed':passed, 'kind':row['kind'], 'finish_reason':finish,
            'completion_tokens':response.get('usage',{}).get('completion_tokens')}
    missing = sorted(set(cases)-set(result))
    return {'cases':result, 'by_kind':dict(counts), 'missing_cases':missing}


def stability(path):
    summary = json.loads((path/'summary.json').read_text())
    telemetry = rows(path/'telemetry.jsonl')
    soak = rows(path/'soak.jsonl') if (path/'soak.jsonl').exists() else []
    scored = rows(path/'quality.jsonl') if (path/'quality.jsonl').exists() else []
    operational = summary['results'][len(scored):]
    ids = {r.get('health',{}).get('instance_id') for r in telemetry}
    ids.update(summary.get(k,{}).get('instance_id') for k in ('initial_health','final_health'))
    cgroups = [r['cgroup'] for r in telemetry if 'cgroup' in r]
    event_counts = [{line.split()[0]:int(line.split()[1]) for line in r['memory.events'].splitlines()} for r in cgroups]
    peak_swap = max((int(r['memory.swap.current']) for r in cgroups), default=None)
    swaps = [r['memory']['SwapTotal']-r['memory']['SwapFree'] for r in telemetry if 'memory' in r]
    return {'duration_seconds':summary['duration_seconds'],
        'duration_gate_passed':summary['duration_seconds'] >= 3600,
        'runtime_checks_passed':not summary['errors'] and bool(operational) and all(r['passed'] for r in operational),
        'errors':summary['errors'],
        'one_healthy_instance':len(ids) == 1 and None not in ids and all(r.get('health',{}).get('status') == 'ok' and 'error' not in r for r in telemetry),
        'instance_ids':sorted(ids, key=str), 'soak_requests':len(soak),
        'scored_requests_during_stability_run':len(scored),
        'all_soak_requests_passed':all(r['passed'] for r in soak),
        'scored_requests_completed_without_transport_error':all('response' in r for r in scored),
        'peak_reported_gpu_reserved_gib':max((r.get('stats',{}).get('vram_bytes',0) for r in telemetry),default=0)/2**30,
        'min_available_host_gib':min((r['memory']['MemAvailable'] for r in telemetry if 'memory' in r),default=0)/2**30,
        'cgroup_peak_swap_bytes':peak_swap,
        'cgroup_oom_events_delta':event_counts[-1]['oom']-event_counts[0]['oom'] if event_counts else None,
        'cgroup_oom_kill_delta':event_counts[-1]['oom_kill']-event_counts[0]['oom_kill'] if event_counts else None,
        'host_swap_growth_bytes':swaps[-1]-swaps[0] if swaps else None,
        'limitation':'Host counters include other processes. CUDA reserved memory is reported by the server; it is distinct from cgroup CPU accounting. Soak throughput is not the fixed SPEED-Bench comparison.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--endurance',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    corpus=json.loads(a.corpus.read_text())
    candidate=quality(a.candidate,corpus); baseline=quality(a.baseline,corpus)
    common=candidate['cases'].keys() & baseline['cases'].keys()
    regressions=sorted(k for k in common if baseline['cases'][k]['passed'] and not candidate['cases'][k]['passed'])
    improvements=sorted(k for k in common if candidate['cases'][k]['passed'] and not baseline['cases'][k]['passed'])
    result={'corpus_sha256':hashlib.sha256(a.corpus.read_bytes()).hexdigest(),
        'candidate':candidate,'baseline':baseline,'introduced_failures':regressions,
        'improvements':improvements,'no_observed_quality_regression':not regressions and not candidate['missing_cases'] and not baseline['missing_cases'],
        'limitation':'A small fixed regression set, not a statistical non-inferiority study. Greedy AIME outcomes with token caps are reported separately from serving correctness.'}
    if a.endurance:
        result['stability']=stability(a.endurance)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','baseline')},indent=2))
