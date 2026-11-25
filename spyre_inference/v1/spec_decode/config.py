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

import torch

MAX_DFLASH_BATCH_SIZE = 4


def normalize_speculative_config(engine_args) -> None:
    spec = engine_args.speculative_config
    if not isinstance(spec, dict):
        return
    method = spec.get("method")
    if method is None:
        model = str(spec.get("model", "")).lower()
        # Match vLLM's name-based detection before it applies cache defaults.
        if "dflash" in model and "eagle-" not in model and "eagle3" not in model:
            method = "dflash"
        elif "dspark" in model:
            method = "dspark"
    if method not in ("dflash", "dspark"):
        return
    engine_args.speculative_config = spec | {"method": method}
    if engine_args.enable_prefix_caching is None:
        engine_args.enable_prefix_caching = False


def _validate_serving_constraints(config, spec, name: str) -> None:
    """What every Spyre draft family requires; `name` scopes the error messages."""
    if spec.max_model_len is None and spec.draft_shares_target_positions():
        # The platform's default length cap must not stop drafts inside the target context.
        spec.draft_model_config.max_model_len = config.model_config.max_model_len
    parallel = config.parallel_config
    if config.model_config.dtype != torch.float16 or spec.draft_model_config.dtype != torch.float16:
        raise ValueError(f"{name} currently requires float16 serving weights")
    if config.model_config.enforce_eager:
        raise ValueError(f"{name} currently requires compiled execution")
    if (
        config.model_config.quantization is not None
        or spec.draft_model_config.quantization is not None
    ):
        raise ValueError(f"{name} currently requires unquantized target and draft weights")
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or not 1 <= config.scheduler_config.max_num_seqs <= MAX_DFLASH_BATCH_SIZE
    ):
        raise ValueError(
            f"{name} currently requires TP=1, PP=1 and max_num_seqs<={MAX_DFLASH_BATCH_SIZE}"
        )
    if config.cache_config.enable_prefix_caching:
        raise ValueError(
            f"Disable prefix caching for {name}: draft context needs every target state"
        )
    target = config.model_config.hf_text_config
    if target.model_type not in {"qwen3", "gemma4", "gemma4_text"} or (
        config.model_config.is_multimodal_model
    ):
        raise ValueError(f"{name} supports Qwen3 and Gemma 4 text targets")
    if spec.draft_sample_method != "greedy":
        raise ValueError(f"{name} currently uses greedy proposals")
    block = spec.num_speculative_tokens + 1
    minimum_budget = config.scheduler_config.max_num_seqs * (
        block + spec.max_num_new_slots_for_drafting
    )
    if config.scheduler_config.max_num_batched_tokens < minimum_budget:
        raise ValueError(
            "max_num_batched_tokens must cover target verification "
            "and vLLM's per-request drafting reservation "
            f"(at least {minimum_budget} tokens)"
        )


def validate_dflash_config(config) -> None:
    spec = config.speculative_config
    if spec is None or not spec.use_dflash():
        return
    draft = spec.draft_model_config.hf_config
    architecture = spec.draft_model_config.architecture
    supported_drafts = {"DFlashDraftModel", "Qwen3XPressModel", "DFlashQwen3XPressModel"}
    if architecture not in supported_drafts:
        raise ValueError(
            f"Unsupported Spyre DFlash/XPress draft architecture: {architecture}. "
            f"Supported architectures: {', '.join(sorted(supported_drafts))}"
        )
    _validate_serving_constraints(config, spec, "Spyre DFlash/XPress")
    target = config.model_config.hf_text_config
    if draft.hidden_size != target.hidden_size or draft.vocab_size != target.vocab_size:
        raise ValueError("The draft must share the target hidden size and full vocabulary")
    if getattr(draft, "draft_vocab_size", draft.vocab_size) not in (None, target.vocab_size):
        raise ValueError("Spyre DFlash/XPress requires the full target vocabulary")
    if getattr(draft, "logit_scale", 1.0) != 1.0 or getattr(draft, "attention_bias", False):
        raise ValueError("Spyre DFlash/XPress requires unscaled logits and no attention bias")
    if draft.dflash_config.get("attention_sink_bias", False):
        raise ValueError("Spyre DFlash/XPress does not support attention sinks")
    layer_types = getattr(draft, "layer_types", None)
    if layer_types is not None and (
        len(layer_types) != draft.num_hidden_layers
        or any(t not in {"full_attention", "sliding_attention"} for t in layer_types)
    ):
        raise ValueError("DFlash requires one full or sliding attention type per draft layer")
    if draft.dflash_config.get("use_swa", False) or "sliding_attention" in (layer_types or []):
        window = draft.dflash_config.get("swa_window_size", getattr(draft, "sliding_window", None))
        if type(window) is not int or window <= 0:
            raise ValueError("DFlash sliding attention requires a positive window size")
    layers = draft.dflash_config.get("target_layer_ids", [])
    if (
        not layers
        or layers != sorted(set(layers))
        or not all(0 <= i < target.num_hidden_layers for i in layers)
    ):
        raise ValueError("The draft checkpoint must identify ordered, distinct target layers")
    block_size = getattr(draft, "xpress_block_size", getattr(draft, "block_size", None))
    if block_size != spec.num_speculative_tokens + 1:
        raise ValueError("num_speculative_tokens must equal the checkpoint block size minus one")
    if architecture in ("Qwen3XPressModel", "DFlashQwen3XPressModel"):
        if not hasattr(draft, "xpress_rank"):
            raise ValueError("The XPress checkpoint is missing its refiner configuration")
        # SpeculativeConfig has already range-checked xpress_topc and written it here.
        if getattr(draft, "xpress_topc", None) is None:
            draft.xpress_topc = 128
        draft.xpress_topc = min(draft.xpress_topc, draft.vocab_size)
        if not 1 <= draft.xpress_topc <= 1024:
            raise ValueError(
                "Spyre XPress requires 1 to 1024 candidates: "
                "min(xpress_topc, vocab_size) must be in this range; "
                "full-vocabulary refinement (xpress_topc=0) is unsupported"
            )
        passes = getattr(draft, "xpress_num_passes", None)
        if passes is None:
            draft.xpress_num_passes = 6
        elif type(passes) is not int or passes < 0:
            raise ValueError("xpress_num_passes must be a nonnegative integer")
    if getattr(draft, "sample_from_anchor", False) or draft.dflash_config.get(
        "sample_from_anchor", False
    ):
        raise ValueError("Spyre XPress requires a fixed anchor in slot zero")


def validate_dspark_config(config) -> None:
    spec = config.speculative_config
    if spec is None or not spec.use_dspark():
        return
    draft = spec.draft_model_config.hf_config
    if spec.draft_model_config.architecture != "Qwen3DSparkModel":
        raise ValueError(
            f"Unsupported Spyre DSpark draft architecture: "
            f"{spec.draft_model_config.architecture}. Supported: Qwen3DSparkModel"
        )
    _validate_serving_constraints(config, spec, "Spyre DSpark")
    target = config.model_config.hf_text_config
    if draft.hidden_size != target.hidden_size or draft.vocab_size != target.vocab_size:
        raise ValueError("The draft must share the target hidden size and full input vocabulary")
    draft_vocab = getattr(draft, "draft_vocab_size", None) or draft.vocab_size
    if not 0 < draft_vocab <= target.vocab_size:
        raise ValueError("draft_vocab_size must be at most the target vocabulary")
    if getattr(draft, "logit_scale", 1.0) != 1.0 or getattr(draft, "attention_bias", False):
        raise ValueError("Spyre DSpark requires unscaled logits and no attention bias")
    # Only speculators-format adapters write an explicit False; the speculator-level
    # default is the anchor layout, which Spyre does not serve.
    if getattr(draft, "sample_from_anchor", True) is not False:
        raise ValueError(
            "Spyre DSpark currently requires the fill-in layout (sample_from_anchor=false)"
        )
    block_size = getattr(draft, "block_size", None)
    if block_size != spec.num_speculative_tokens + 1:
        raise ValueError(
            "num_speculative_tokens must equal the checkpoint block size minus one; "
            "vLLM defaults it to block_size, so pass "
            f"num_speculative_tokens={(block_size or 0) - 1} explicitly"
        )
    aux_layers = getattr(draft, "eagle_aux_hidden_state_layer_ids", None)
    if (
        not aux_layers
        or aux_layers != sorted(set(aux_layers))
        or not all(1 <= i < target.num_hidden_layers for i in aux_layers)
    ):
        raise ValueError(
            "The draft checkpoint must identify ordered, distinct aux hidden-state "
            "layers inside the target (layer 0 has no predecessor to project from)"
        )
    rank = getattr(draft, "markov_rank", None)
    if type(rank) is not int or rank <= 0:
        raise ValueError("Spyre DSpark requires a positive integer markov_rank")
    if getattr(spec, "dspark_draft_topk", None) or getattr(draft, "dspark_draft_topk", None):
        raise ValueError("Spyre DSpark does not support dspark_draft_topk")
    if getattr(spec, "enable_adaptive_verification", False):
        raise ValueError("Spyre DSpark does not support adaptive verification")
    # The confidence head only feeds adaptive verification; skip its fp32 weights.
    draft.enable_confidence_head = False
