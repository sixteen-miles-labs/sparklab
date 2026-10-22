"""Handwritten factual and usage controls for the independent recall collector."""
import copy
import json

import pytest

from benchmarks import validate_prefix_recall as recall


def report():
    answers = [
        {"fact":"k0", "value":"AX-4317", "source":"P001"},
        {"fact":"k1", "value":"2031-06-19", "source":"P500"},
        {"fact":"k2", "value":7341, "source":"P999"},
    ]
    cases = []
    for answer in answers:
        msg = {"role":"assistant", "content":json.dumps(answer)}
        for phase in ("seed", "repeat"):
            cases.append({"length":8192, "key":answer["fact"], "phase":phase, "validated":True, "error":None,
                "rendered":{"prompt_tokens":8192,"token_ids_sha256":"a"*64},
                "response":{"choices":[{"message":msg,"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":8192,"completion_tokens":37,"total_tokens":8229}},
                "output_signature":{"content":msg["content"],"reasoning_content":None,"tool_calls":[]}})
    return {"complete":True,"identity":{"artifact":"fixed"},
        "workload":{"lengths":[8192],"output_budget":256},"cases":cases}


def test_complete_control_compares_without_relying_on_cache_counter():
    a = report(); b = copy.deepcopy(a)
    b["cases"][1]["response"]["usage"]["prompt_tokens_details"]={"cached_tokens":8100}
    recall.compare(a,b)


@pytest.mark.parametrize("fault", ["answer","type","truncated","usage","cached","duplicate","signature","render","identity","empty","aggregate-error"])
def test_invalid_answers_incomplete_work_and_mismatched_controls_fail(fault):
    a = report(); b = copy.deepcopy(a); c = b["cases"][0]
    if fault=="answer": c["response"]["choices"][0]["message"]["content"]='{"fact":"k0","value":"wrong","source":"P001"}'
    elif fault=="type": c["response"]["usage"]["completion_tokens"]=True
    elif fault=="truncated": c["response"]["choices"][0]["finish_reason"]="length"
    elif fault=="usage": c["response"]["usage"]["prompt_tokens"]=8191
    elif fault=="cached": c["response"]["usage"]["prompt_tokens_details"]={"cached_tokens":8193}
    elif fault=="duplicate": b["cases"][1]=copy.deepcopy(c)
    elif fault=="signature": c["output_signature"]["content"]="different"
    elif fault=="render": c["rendered"]["token_ids_sha256"]="b"*64
    elif fault=="empty": b["workload"]["lengths"]=[]; b["cases"]=[]
    elif fault=="aggregate-error": b["error"]="previous comparison failed"
    else: b["identity"]["artifact"]="other"
    with pytest.raises(ValueError): recall.compare(a,b)


def test_exact_hit_receipt_cannot_be_replaced_by_success_or_page_aligned_reuse():
    a = report(); b = copy.deepcopy(a)
    b["requires_exact_hits"] = True
    b["native_stats"] = {"requests":{"active":0},"native_runtime":{"exact_prefix":{"published":6,"hits":3}}}
    for case in b["cases"]:
        if case["phase"]=="repeat":
            case["response"]["usage"]["prompt_tokens_details"]={"cached_tokens":8191}
    recall.compare(a,b)
    b["cases"][1]["response"]["usage"]["prompt_tokens_details"]["cached_tokens"]=8176
    with pytest.raises(ValueError, match="exact unaligned"):
        recall.compare(a,b)
