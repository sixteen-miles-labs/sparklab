import mmap

import pytest
import torch

from freetoken.moe.host_banks import HostBank


class _FakeBuffer:
    def __init__(self, *, reject_remove=False):
        self.reject_remove = reject_remove
        self.advice = []

    def madvise(self, advice):
        self.advice.append(advice)
        if self.reject_remove and advice == mmap.MADV_REMOVE:
            raise OSError("MADV_REMOVE unsupported")


def _bank_with_buffer(buf):
    bank = HostBank.__new__(HostBank)
    bank.tensor = torch.empty(0)
    bank.addr = 0
    bank.nbytes = 0
    bank._buf = buf
    bank._pinned = False
    return bank


@pytest.mark.skipif(not hasattr(mmap, "MADV_REMOVE"), reason="MADV_REMOVE unavailable")
def test_release_removes_shared_anonymous_pages_immediately():
    buf = _FakeBuffer()
    _bank_with_buffer(buf).release()
    assert buf.advice == [mmap.MADV_REMOVE]


@pytest.mark.skipif(not hasattr(mmap, "MADV_REMOVE"), reason="MADV_REMOVE unavailable")
def test_release_falls_back_when_remove_is_unsupported():
    buf = _FakeBuffer(reject_remove=True)
    _bank_with_buffer(buf).release()
    assert buf.advice == [mmap.MADV_REMOVE, mmap.MADV_DONTNEED]
