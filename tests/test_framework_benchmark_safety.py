from pathlib import Path
import importlib.util
import json
import sys


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "frameworks" / "run_framework.py"
SPEC = importlib.util.spec_from_file_location("run_framework_safety", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sglang_is_limited_to_batch_one_cache():
    command, _ = MODULE.command_for("sglang", Path("/model"), Path("/model.gguf"), 18080)

    assert command[command.index("--max-running-requests") + 1] == "1"
    assert command[command.index("--max-total-tokens") + 1] == "32768"
    assert command[command.index("--max-mamba-cache-size") + 1] == "5"
    assert command[command.index("--watchdog-timeout") + 1] == "1200"


def test_sglang_disables_parallel_tuning(monkeypatch):
    for name in (
        "SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE",
        "MAX_JOBS",
        "FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS",
    ):
        monkeypatch.delenv(name, raising=False)

    environment = MODULE.environment_for("sglang")

    assert environment["SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE"] == "0"
    assert environment["MAX_JOBS"] == "4"
    assert environment["FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS"] == "1"


def test_checkpoint_must_contain_nvfp4_metadata(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({
        "quantization_config": {
            "quantized_layers": {"model.layers.0.mlp": {"quant_algo": "FP8"}}
        }
    }))

    try:
        MODULE.validate_nvfp4_checkpoint(tmp_path)
    except ValueError as error:
        assert "not the required ModelOpt NVFP4" in str(error)
    else:
        raise AssertionError("an FP8 checkpoint must not pass NVFP4 validation")


def test_native_weight_frameworks_are_not_misclassified_as_nvfp4():
    for framework in ("llama.cpp", "ktransformers", "ollama"):
        assert framework not in MODULE.NVFP4_FRAMEWORKS


def test_native_framework_commands_are_serialized():
    llama, _ = MODULE.command_for("llama.cpp", Path("/model"), Path("/model.gguf"), 18080)
    ollama, _ = MODULE.command_for("ollama", Path("/model"), Path("/model.gguf"), 18080)

    assert llama[llama.index("--parallel") + 1] == "1"
    assert llama[llama.index("-c") + 1] == "32768"
    assert ollama[-1] == "serve"


def test_ollama_control_disables_mtp_and_uses_gpu():
    command, _ = MODULE.command_for(
        "ollama-no-mtp", Path("/model"), Path("/model.gguf"), 18080
    )

    assert "--spec-type" not in command
    assert command[command.index("-ngl") + 1] == "99"


def test_freetoken_and_sparklab_use_identical_runtime_flags():
    ftw = Path("/model.ftw")
    freetoken, _ = MODULE.command_for("freetoken", Path("/hf"), Path("/gguf"), 18080, ftw)
    sparklab, _ = MODULE.command_for("sparklab", Path("/hf"), Path("/gguf"), 18080, ftw)

    assert freetoken[1:] == sparklab[1:]
    assert freetoken[freetoken.index("--moe-cache-rate") + 1] == "1.0"
    assert freetoken[freetoken.index("--max-running-requests") + 1] == "1"


def test_qwen38_uses_each_runtime_supported_qsa_path():
    ftw = Path("/qwen38.ftw")
    freetoken, _ = MODULE.command_for(
        "freetoken", Path("/hf"), Path("/gguf"), 18080, ftw, "qwen38"
    )
    sparklab, _ = MODULE.command_for(
        "sparklab", Path("/hf"), Path("/gguf"), 18080, ftw, "qwen38"
    )

    assert freetoken[freetoken.index("--attention-backend") + 1] == "qsa_sparse"
    assert sparklab[sparklab.index("--attention-backend") + 1] == "qsa"
    assert "--moe-preload-all" not in freetoken
    assert "--moe-preload-all" in sparklab
    assert sparklab[sparklab.index("--moe-storage") + 1] == "disk"


def test_memory_watchdog_detects_low_available_memory(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemAvailable:    8388608 kB\n"
        "SwapTotal:      16777216 kB\n"
        "SwapFree:       15728640 kB\n"
    )

    reason = MODULE.memory_safety_reason(
        min_available_gib=12, min_swap_free_gib=2, meminfo=meminfo
    )

    assert reason == "MemAvailable 8.0 GiB fell below 12.0 GiB"


def test_memory_watchdog_detects_low_swap(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemAvailable:   67108864 kB\n"
        "SwapTotal:      16777216 kB\n"
        "SwapFree:        1048576 kB\n"
    )

    reason = MODULE.memory_safety_reason(
        min_available_gib=12, min_swap_free_gib=2, meminfo=meminfo
    )

    assert reason == "SwapFree 1.0 GiB fell below 2.0 GiB"
