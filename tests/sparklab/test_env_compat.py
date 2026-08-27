from __future__ import annotations

from freetoken.env_compat import getenv_compat, product_name


def test_product_environment_wins_with_legacy_fallback(monkeypatch):
    monkeypatch.setenv("FREETOKEN_EXAMPLE", "legacy")
    assert getenv_compat("FREETOKEN_EXAMPLE") == "legacy"
    monkeypatch.setenv("SPARKLAB_EXAMPLE", "product")
    assert getenv_compat("FREETOKEN_EXAMPLE") == "product"
    assert product_name("FREETOKEN_EXAMPLE") == "SPARKLAB_EXAMPLE"


def test_bandwidth_profile_writes_new_path_and_discovers_legacy(tmp_path, monkeypatch):
    import json

    from freetoken.moe import bench_profile

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    new_path = tmp_path / "sparklab" / "benchbw.json"
    legacy_path = tmp_path / "freetoken" / "benchbw.json"
    legacy_path.parent.mkdir()
    legacy_path.write_text(
        json.dumps({"gpu": {"name": "NVIDIA GB10"}, "dtypes": {"nvfp4": "offload"}})
    )

    assert bench_profile.default_profile_path() == str(new_path)
    assert bench_profile.load_backend_recommendation("nvfp4", "NVIDIA GB10") == "offload"

    new_path.parent.mkdir()
    new_path.write_text(
        json.dumps({"gpu": {"name": "NVIDIA GB10"}, "dtypes": {"nvfp4": "hybrid"}})
    )
    assert bench_profile.load_backend_recommendation("nvfp4", "NVIDIA GB10") == "hybrid"
