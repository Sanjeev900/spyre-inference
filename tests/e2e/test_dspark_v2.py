# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DSpark on the V2 runner: greedy output matches the target alone, with no compile
after warmup and no CPU fallback."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]

PROMPTS = [
    "Continue counting by ones, separated by commas:\n1, 2, 3, 4, 5,",
    "2, 4, 6, 8,",
    "Read the following context. "
    + "IBM develops computers, software, and consulting services. " * 8
    + "\nName the company described above:",
    "What are the main businesses of IBM? Answer in one sentence.",
    "Count by fives, separated by commas:\n5, 10, 15,",
    "3, 6, 9, 12,",
    "Read the following context. "
    + "The Spyre accelerator runs large language models efficiently. " * 6
    + "\nName the hardware described above:",
    "Which company builds the Spyre accelerator? Answer in one sentence.",
]


def _number_list(count: int) -> str:
    return ", ".join(str(i) for i in range(1, count + 1)) + ","


# The fill-in layout drafts 1 + K = block_size = 7 rows per request, so each
# ladder covers the draft blocks (7 rows x 1/2/4/8 requests, up to the cap)
# plus the prefill chunk width.
CASES = {
    "gemma4": dict(
        target="google/gemma-4-26B-A4B-it",
        target_revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        draft="makora-ai/gemma4-26b-a4b-dspark",
        draft_revision="e6f739231e892fe4512c470136490d5441ffe62f",
        engine=dict(
            max_model_len=8192,
            max_num_seqs=8,
            max_num_batched_tokens=128,
            compilation_config={"compile_sizes": [1, 7, 14, 28, 56, 128]},
            gemma4_text_backbone=True,
        ),
        env={
            "SPYRE_ATTN_QUERY_BUCKETS": "1,7,14,28,56,128",
            "SPYRE_ATTN_KV_BUCKETS": "128,256,8192",
        },
        batches={
            # Different lengths, so the active draft batch shrinks through 8, 4, 2 and 1.
            "batch": (PROMPTS, [48, 40, 24, 32, 36, 28, 20, 16]),
        },
    ),
    "gemma4_tp4": dict(
        target="google/gemma-4-26B-A4B-it",
        target_revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        draft="makora-ai/gemma4-26b-a4b-dspark",
        draft_revision="e6f739231e892fe4512c470136490d5441ffe62f",
        engine=dict(
            max_model_len=32768,
            max_num_seqs=32,
            max_num_batched_tokens=224,
            compilation_config={"compile_sizes": [1, 7, 14, 28, 56, 112, 224]},
            tensor_parallel_size=4,
            gemma4_text_backbone=True,
        ),
        env={
            "SPYRE_ATTN_QUERY_BUCKETS": "1,7,14,28,56,112,224",
            "SPYRE_ATTN_KV_BUCKETS": "128,1024,4096,16384,32768",
        },
        batches={
            "batch": (PROMPTS, [48, 40, 24, 32, 36, 28, 20, 16]),
            # A prompt near the context cap, where drafting must stop cleanly.
            "very_long": ([_number_list(6500)], [16]),
        },
    ),
}


def _engine_outputs(case_name: str, speculative: bool) -> dict:
    from vllm import LLM, SamplingParams

    from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

    case = CASES[case_name]
    engine = dict(case["engine"])
    if engine.pop("gemma4_text_backbone", False):
        engine["hf_overrides"] = GEMMA4_TEXT_BACKBONE_OVERRIDE
    spec = dict(
        method="dspark",
        model=case["draft"],
        revision=case["draft_revision"],
        # vLLM defaults K to block_size (7); the fill-in layout needs block_size - 1.
        num_speculative_tokens=6,
    )
    llm = LLM(
        model=case["target"],
        revision=case["target_revision"],
        tokenizer_revision=case["target_revision"],
        dtype="float16",
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config=spec if speculative else None,
        **engine,
    )
    outputs = {}
    for name, (prompts, limits) in case["batches"].items():
        params = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in limits]
        results = llm.generate(prompts, params)
        outputs[name] = [list(r.outputs[0].token_ids) for r in results]
    metrics = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
    outputs["drafted"] = metrics.get("vllm:spec_decode_num_draft_tokens")
    outputs["accepted"] = metrics.get("vllm:spec_decode_num_accepted_tokens")
    # Frees the card for the next engine in this process.
    llm.llm_engine.engine_core.shutdown()
    return outputs


def _both_engines(case_name: str) -> dict:
    """Runs in the child: the Triton gate binds the runner at vLLM import."""
    import warnings

    import torch  # noqa: F401  (autoloads torch_spyre)
    from torch_spyre.ops.fallbacks import FallbackWarning

    warnings.simplefilter("error", FallbackWarning)
    reference = _engine_outputs(case_name, speculative=False)
    return {"reference": reference, "spec": _engine_outputs(case_name, speculative=True)}


def _generate(case_name: str) -> dict:
    env = {
        **os.environ,
        "SPYRE_ATTN_KV_LAYOUT": "head_major",
        "SPYRE_COMPILE_GUARD": "error",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "36000",
        **CASES[case_name]["env"],
    }
    result = subprocess.run(
        [sys.executable, __file__, case_name],
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert result.returncode == 0 and lines, result.stdout[-4000:] + result.stderr[-6000:]
    return json.loads(lines[-1].removeprefix("RESULT "))


@pytest.mark.timeout(7200)
# Multi-card cases carry the 4-card mark so the dedicated spyre_pf_x4 CI job
# collects them; the 1-card quality shards collect them and skip on device count.
@pytest.mark.parametrize(
    "case_name",
    [
        pytest.param(name, marks=pytest.mark.distributed_tp4)
        if "tensor_parallel_size" in CASES[name]["engine"]
        else name
        for name in CASES
    ],
)
def test_dspark_on_v2_matches_the_target_alone(case_name):
    from spyre_testing_plugin.pytest_plugin import spyre_device_count, spyre_hardware_present

    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    if CASES[case_name]["engine"].get("tensor_parallel_size", 1) > spyre_device_count():
        pytest.skip("Requires more Spyre accelerators than this host has")
    result = _generate(case_name)
    reference, speculative = result["reference"], result["spec"]
    for name in CASES[case_name]["batches"]:
        assert speculative[name] == reference[name], name
    assert speculative["drafted"] and speculative["accepted"] > 0, speculative


if __name__ == "__main__":
    print("RESULT " + json.dumps(_both_engines(sys.argv[1])))
