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

"""validate_dspark_config and normalize_speculative_config for DSpark.

The accepted fixture is the makora-ai/gemma4-26b-a4b-dspark contract after
vLLM's speculators adapter rewrote it.
"""

from types import SimpleNamespace as NS

import pytest
import torch

from spyre_inference.v1.spec_decode.config import (
    normalize_speculative_config,
    validate_dspark_config,
)


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"model": "draft"},
        {"model": "eagle-DFlash"},
        {"model": "eagle3-head", "method": "eagle3"},
    ],
)
def test_dspark_inference_preserves_other_methods(spec):
    args = NS(speculative_config=spec, enable_prefix_caching=None)
    normalize_speculative_config(args)
    assert args.speculative_config is spec
    assert args.enable_prefix_caching is None


def test_a_dspark_draft_name_infers_the_method_and_disables_prefix_caching():
    args = NS(
        speculative_config={"model": "makora-ai/gemma4-26b-a4b-dspark"},
        enable_prefix_caching=None,
    )
    normalize_speculative_config(args)
    assert args.speculative_config["method"] == "dspark"
    assert args.enable_prefix_caching is False


@pytest.fixture
def config():
    target = NS(model_type="gemma4", hidden_size=2816, vocab_size=262144, num_hidden_layers=30)
    draft = NS(
        hidden_size=2816,
        vocab_size=262144,
        num_hidden_layers=5,
        layer_types=["full_attention"] * 5,
        draft_vocab_size=32000,
        markov_rank=256,
        block_size=7,
        sample_from_anchor=False,
        eagle_aux_hidden_state_layer_ids=[3, 10, 18, 25, 28],
        logit_scale=1.0,
        attention_bias=False,
        enable_confidence_head=True,
    )
    return NS(
        speculative_config=NS(
            use_dspark=lambda: True,
            use_dflash=lambda: False,
            draft_shares_target_positions=lambda: True,
            max_model_len=None,
            draft_model_config=NS(
                hf_config=draft,
                architecture="Qwen3DSparkModel",
                quantization=None,
                dtype=torch.float16,
                max_model_len=2048,
            ),
            num_speculative_tokens=6,
            max_num_new_slots_for_drafting=0,
            draft_sample_method="greedy",
            dspark_draft_topk=None,
            enable_adaptive_verification=False,
        ),
        model_config=NS(
            hf_text_config=target,
            max_model_len=2048,
            is_multimodal_model=False,
            dtype=torch.float16,
            enforce_eager=False,
            quantization=None,
        ),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=64),
        cache_config=NS(enable_prefix_caching=False),
    )


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_the_makora_contract_is_accepted_and_the_confidence_head_is_dropped(config, batch):
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = 28 * batch
    validate_dspark_config(config)
    draft = config.speculative_config.draft_model_config.hf_config
    assert draft.enable_confidence_head is False


def test_the_draft_inherits_the_target_context_length(config):
    validate_dspark_config(config)
    assert config.speculative_config.draft_model_config.max_model_len == 2048


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("architecture", "DSparkDraftModel", "Unsupported Spyre DSpark draft architecture"),
        ("hidden_size", 4096, "share the target hidden size"),
        ("draft_vocab_size", 300000, "at most the target vocabulary"),
        ("logit_scale", 2.0, "unscaled logits"),
        ("sample_from_anchor", True, "fill-in layout"),
        ("block_size", 8, "block size minus one"),
        ("eagle_aux_hidden_state_layer_ids", [0, 10], "aux hidden-state"),
        ("eagle_aux_hidden_state_layer_ids", [30, 10], "aux hidden-state"),
        ("markov_rank", 0, "positive integer markov_rank"),
    ],
)
def test_contract_violations_fail(config, name, value, match):
    draft = config.speculative_config.draft_model_config
    if name == "architecture":
        draft.architecture = value
    else:
        setattr(draft.hf_config, name, value)
    with pytest.raises(ValueError, match=match):
        validate_dspark_config(config)


def test_an_absent_sample_from_anchor_is_the_anchor_layout_and_fails(config):
    draft = config.speculative_config.draft_model_config.hf_config
    del draft.sample_from_anchor
    with pytest.raises(ValueError, match="fill-in layout"):
        validate_dspark_config(config)


def test_the_block_size_trap_names_the_required_tokens(config):
    # vLLM defaults num_speculative_tokens to block_size (7), not block_size - 1.
    config.speculative_config.num_speculative_tokens = 7
    with pytest.raises(ValueError, match="pass num_speculative_tokens=6 explicitly"):
        validate_dspark_config(config)


def test_topk_and_adaptive_verification_are_rejected(config):
    config.speculative_config.dspark_draft_topk = 16
    with pytest.raises(ValueError, match="dspark_draft_topk"):
        validate_dspark_config(config)
    config.speculative_config.dspark_draft_topk = None
    config.speculative_config.enable_adaptive_verification = True
    with pytest.raises(ValueError, match="adaptive verification"):
        validate_dspark_config(config)


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("dtype", torch.bfloat16, "float16 serving weights"),
        ("enforce_eager", True, "compiled execution"),
        ("quantization", "fp8", "unquantized"),
    ],
)
def test_shared_serving_constraints_apply(config, name, value, match):
    setattr(config.model_config, name, value)
    with pytest.raises(ValueError, match=match):
        validate_dspark_config(config)


def test_shared_parallel_and_batch_limits_apply(config):
    config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="TP=1, PP=1"):
        validate_dspark_config(config)
    config.parallel_config.tensor_parallel_size = 1
    config.scheduler_config.max_num_seqs = 5
    with pytest.raises(ValueError, match="max_num_seqs"):
        validate_dspark_config(config)


def test_the_token_budget_covers_every_verified_row(config):
    config.scheduler_config.max_num_batched_tokens = 6
    with pytest.raises(ValueError, match="at least 7 tokens"):
        validate_dspark_config(config)


def test_dflash_configs_do_not_reach_the_dspark_validator(config):
    config.speculative_config.use_dspark = lambda: False
    draft = config.speculative_config.draft_model_config.hf_config
    draft.block_size = 99
    validate_dspark_config(config)
