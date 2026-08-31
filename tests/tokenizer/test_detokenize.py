from sparklab.message import DetokenizeMsg
from sparklab.tokenizer.detokenize import DetokenizeManager


class _Tokenizer:
    eos_token_id = 0

    def batch_decode(self, rows):
        pieces = {1: "first\n", 2: "second\n"}
        return ["".join(pieces[token] for token in row) for row in rows]


def test_multiple_tokens_for_one_request_are_detokenized_sequentially():
    manager = DetokenizeManager(_Tokenizer())
    messages = [
        DetokenizeMsg(uid=7, next_token=1, finished=False),
        DetokenizeMsg(uid=7, next_token=2, finished=False),
    ]

    assert manager.detokenize(messages) == ["first\n", "second\n"]
    assert manager.decode_map[7].decoded_ids == [1, 2]
