"""Compare native next-token log-probabilities to Prism on identical token IDs."""

import argparse
import json
from pathlib import Path

import requests
import torch

PROMPTS = [
    "The capital of France is",
    "17 * 23 =",
    "def square(x):\n    return",
    "法国的首都是",
    "<|im_start|>user\nExplain how a rainbow forms.<|im_end|>\n<|im_start|>assistant\n<think>\n",
]


def run(a):
    from sparklab.models.gguf.tokenizer import load_gguf_tokenizer

    tok = load_gguf_tokenizer(str(a.model))
    rows = []
    for prompt in PROMPTS:
        ids = requests.post(
            a.reference + "/tokenize", json={"content": prompt}, timeout=30
        ).json()["tokens"]
        assert tok.encode(prompt, add_special_tokens=False) == ids
        a.capture.with_suffix(".request").touch()
        native = requests.post(
            a.native + "/v1/completions",
            json={
                "model": "bonsai2",
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 1,
            },
            timeout=120,
        )
        native.raise_for_status()
        logits = torch.load(a.capture.with_suffix(".pt"), weights_only=True).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite native logits")
        lp = logits.log_softmax(-1)
        ref = requests.post(
            a.reference + "/completion",
            json={
                "prompt": ids,
                "n_predict": 1,
                "temperature": 0,
                "n_probs": 100,
                "post_sampling_probs": False,
                "cache_prompt": False,
            },
            timeout=120,
        )
        ref.raise_for_status()
        probs = ref.json()["completion_probabilities"][0]
        top = probs["top_logprobs"]
        errors = [abs(float(lp[x["id"]]) - x["logprob"]) for x in top]
        row = {
            "prompt": prompt,
            "tokens": ids,
            "native_argmax": int(lp.argmax()),
            "reference_argmax": probs["id"],
            "argmax_match": int(lp.argmax()) == probs["id"],
            "top100_logprob_mae": sum(errors) / len(errors),
            "top100_logprob_max_error": max(errors),
            "reference_top5": top[:5],
            "native_top5": [
                {"id": int(i), "logprob": float(lp[i])} for i in lp.topk(5).indices
            ],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.output.write_text(json.dumps({"cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--native", default="http://127.0.0.1:8093")
    p.add_argument("--reference", default="http://127.0.0.1:8092")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--capture", type=Path, default=Path("/tmp/bonsai2-capture"))
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
