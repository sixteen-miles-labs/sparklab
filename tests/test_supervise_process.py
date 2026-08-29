from benchmarks.supervise_process import _cgroup_safety_reason, _safety_reason


def _reason(current_vm=None, swap_current=100):
    return _safety_reason(
        {"oom_kill": 7, "pswpin": 11, "pswpout": 13},
        current_vm or {"oom_kill": 7, "pswpin": 11, "pswpout": 13},
        100,
        swap_current,
        max_oom_kill_delta=0,
        max_pswpout_delta=0,
        max_swap_growth_bytes=0,
    )


def test_safety_reason_accepts_zero_deltas():
    assert _reason() is None


def test_safety_reason_rejects_oom_delta():
    assert _reason({"oom_kill": 8, "pswpin": 11, "pswpout": 13}) == (
        "oom_kill delta 1 exceeded 0"
    )


def test_safety_reason_rejects_swap_out_delta():
    assert _reason({"oom_kill": 7, "pswpin": 11, "pswpout": 14}) == (
        "pswpout delta 1 exceeded 0"
    )


def test_safety_reason_rejects_swap_growth():
    assert _reason(swap_current=101) == "swap growth 1 bytes exceeded 0 bytes"


def _cgroup(*, swap_max="0", oom_kill=0, swap_current=0):
    return {
        "swap_max": swap_max,
        "swap_current_bytes": swap_current,
        "events_local": {"oom_kill": oom_kill},
    }


def test_cgroup_safety_accepts_no_swap_isolation():
    assert _cgroup_safety_reason(
        _cgroup(),
        _cgroup(),
        require_swap_disabled=True,
        max_oom_kill_delta=0,
        max_swap_growth_bytes=0,
    ) is None


def test_cgroup_safety_requires_swap_disabled():
    assert _cgroup_safety_reason(
        _cgroup(),
        _cgroup(swap_max="max"),
        require_swap_disabled=True,
        max_oom_kill_delta=0,
        max_swap_growth_bytes=0,
    ) == "cgroup memory.swap.max='max', expected '0'"


def test_cgroup_safety_rejects_local_oom():
    assert _cgroup_safety_reason(
        _cgroup(),
        _cgroup(oom_kill=1),
        require_swap_disabled=True,
        max_oom_kill_delta=0,
        max_swap_growth_bytes=0,
    ) == "cgroup oom_kill delta 1 exceeded 0"


def test_cgroup_safety_rejects_swap_growth():
    assert _cgroup_safety_reason(
        _cgroup(),
        _cgroup(swap_current=1),
        require_swap_disabled=True,
        max_oom_kill_delta=0,
        max_swap_growth_bytes=0,
    ) == "cgroup swap growth 1 bytes exceeded 0 bytes"
