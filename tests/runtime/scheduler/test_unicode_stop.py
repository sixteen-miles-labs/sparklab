"""Real byte-level tokenizer and production drain/codec/detokenizer stop checks."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from sparklab.core import Batch
from sparklab.message import BaseTokenizerMsg
from sparklab.runtime.scheduler.scheduler import Scheduler
from sparklab.tokenizer.detokenize import DetokenizeManager
from tests.runtime.scheduler.test_abort_inflight_prefill import _setup, _launch_req


def byte_tokenizer():
    # Every UTF-8 byte gets its own token. The tokenizer's real ByteLevel decoder
    # independently reconstructs Unicode; tests do not emulate scheduler matching.
    vocab = {
        value: i for i, value in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


@pytest.mark.parametrize("stop", ["🙂", "界", "é", "🙂界", "END"])
def test_stop_matches_at_exact_utf8_completion_with_bounded_lookback(stop):
    tokenizer = byte_tokenizer()
    prefix = tokenizer.encode("a" * 1024).ids
    ids = tokenizer.encode(stop).ids
    assert len(ids) == len(stop.encode("utf-8"))
    windows = []

    def decode(tokens):
        windows.append(len(tokens))
        return tokenizer.decode(tokens)

    owner = SimpleNamespace(tokenizer=SimpleNamespace(decode=decode))
    prompt = [7, 8]
    request = SimpleNamespace(
        max_device_len=4098,
        output_len=4096,
        sampling_params=SimpleNamespace(stop_strs=[stop]),
    )
    for count in range(1, len(ids) + 1):
        request.input_ids = torch.tensor(
            prompt + prefix + ids[:count], dtype=torch.int32
        )
        match = Scheduler._match_stop_str(owner, request)
        assert match == (stop if count == len(ids) else None)
    assert max(windows) <= len(stop.encode("utf-8")) + 1


@pytest.mark.parametrize("stop", ["🙂", "界", "é", "🙂界", "END"])
def test_speculative_burst_stops_before_suffix_and_detokenizes_without_leak(stop):
    tokenizer = byte_tokenizer()
    generated = tokenizer.encode("safe " + stop + "LEAK").ids
    expected_tokens = tokenizer.encode("safe " + stop).ids
    pool, cm, tm, dm, _, sent, owner = _setup()
    owner.tokenizer = tokenizer
    owner._match_stop_str = lambda req: Scheduler._match_stop_str(owner, req)
    req = _launch_req(
        pool, cm, tm, torch.tensor([7, 8], dtype=torch.int32), output_len=32
    )
    req.sampling_params.stop_strs = [stop]
    dm.filter_reqs([req])
    # Simulate a verified burst already committed beyond an early stop. The real
    # host drain must stop shipping tokens and refuse overadvanced state donation.
    req.device_len = 2 + len(generated) - 1
    cm.allocate_paged([req])
    req.complete_one()
    try:
        output = SimpleNamespace(
            next_tokens_cpu=torch.tensor(generated, dtype=torch.int32),
            token_counts=(len(generated),),
            copy_done_event=SimpleNamespace(synchronize=lambda: None),
        )
        Scheduler._process_last_data(
            owner, (SimpleNamespace(batch=Batch([req], "verify")), output)
        )
        assert [msg.next_token for msg in sent] == expected_tokens
        assert sent[-1].finished and sent[-1].finish_reason == "stop"
        assert sent[-1].matched_stop == stop
        assert not any(msg.finished for msg in sent[:-1])
        assert req.state_overadvanced and req.table_idx == -1
        assert req not in dm.running_reqs
        replies = [
            BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg)) for msg in sent
        ]
        decoder = DetokenizeManager(
            SimpleNamespace(
                eos_token_id=-1,
                batch_decode=lambda rows: [tokenizer.decode(row) for row in rows],
            )
        )
        assert "".join(decoder.detokenize(replies)) == "safe "
        assert not decoder.decode_map
        cm.check_integrity()
        assert pool.num_free_slots == 15
    finally:
        if req.table_idx != -1:
            req.aborted = True
            owner._free_req_resources(req)
        owner._free_req_resources = None  # break the synthetic owner's capture cycle
