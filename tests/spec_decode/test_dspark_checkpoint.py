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

"""validate_dspark_weights against the makora-ai/gemma4-26b-a4b-dspark layout."""

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.spec_decode.checkpoint import validate_dspark_weights

# The checkpoint's own numbers: 66 tensors, five full-attention layers at the
# target's text width, a 32k draft vocabulary over the 262k target vocabulary.
HIDDEN, HEAD, KV_HEADS, RANK = 2816, 256, 8, 256
VOCAB, DRAFT_VOCAB, INTERMEDIATE, LAYERS = 262144, 32000, 2112, 5
QUERY = 16 * HEAD
AUX_LAYERS = [3, 10, 18, 25, 28]

LAYER_SHAPES = {
    "self_attn.q_proj.weight": (QUERY, HIDDEN),
    "self_attn.k_proj.weight": (KV_HEADS * HEAD, HIDDEN),
    "self_attn.v_proj.weight": (KV_HEADS * HEAD, HIDDEN),
    "self_attn.o_proj.weight": (HIDDEN, QUERY),
    "self_attn.q_norm.weight": (HEAD,),
    "self_attn.k_norm.weight": (HEAD,),
    "mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
    "mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
    "mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
    "input_layernorm.weight": (HIDDEN,),
    "post_attention_layernorm.weight": (HIDDEN,),
}


def _config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        head_dim=HEAD,
        num_attention_heads=16,
        num_key_value_heads=KV_HEADS,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        vocab_size=VOCAB,
        draft_vocab_size=DRAFT_VOCAB,
        markov_rank=RANK,
        eagle_aux_hidden_state_layer_ids=AUX_LAYERS,
        target_hidden_size=None,
        attention_bias=False,
    )


def _weights():
    shapes = {
        "fc.weight": (HIDDEN, HIDDEN * len(AUX_LAYERS)),
        "hidden_norm.weight": (HIDDEN,),
        "norm.weight": (HIDDEN,),
        "embed_tokens.weight": (VOCAB, HIDDEN),
        "lm_head.weight": (DRAFT_VOCAB, HIDDEN),
        "markov_head.markov_w1.weight": (VOCAB, RANK),
        "markov_head.markov_w2.weight": (DRAFT_VOCAB, RANK),
        "d2t": (DRAFT_VOCAB,),
        "t2d": (VOCAB,),
        "confidence_head.proj.weight": (1, HIDDEN + RANK),
        "confidence_head.proj.bias": (1,),
    }
    for layer in range(LAYERS):
        shapes.update({f"layers.{layer}.{n}": s for n, s in LAYER_SHAPES.items()})
    return {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}


def test_the_makora_layout_is_complete_and_confidence_weights_are_ignored():
    validate_dspark_weights(_config(), _weights())


@pytest.mark.parametrize(
    "name",
    [
        "fc.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "d2t",
        "lm_head.weight",
        "layers.3.self_attn.k_proj.weight",
    ],
)
def test_missing_weight_fails(name):
    weights = _weights()
    del weights[name]
    with pytest.raises(ValueError, match="missing="):
        validate_dspark_weights(_config(), weights)


def test_extra_weight_fails():
    weights = _weights()
    weights["markov_head.markov_w3.weight"] = torch.empty(1, device="meta")
    with pytest.raises(ValueError, match="unexpected="):
        validate_dspark_weights(_config(), weights)


@pytest.mark.parametrize(
    "name,shape",
    [
        ("fc.weight", (HIDDEN, HIDDEN * 4)),
        ("lm_head.weight", (VOCAB, HIDDEN)),
        ("markov_head.markov_w2.weight", (DRAFT_VOCAB, RANK * 2)),
        ("d2t", (DRAFT_VOCAB + 1,)),
    ],
)
def test_shape_mismatch_fails(name, shape):
    weights = _weights()
    weights[name] = torch.empty(shape, device="meta")
    with pytest.raises(ValueError, match=f"Invalid {name} shape"):
        validate_dspark_weights(_config(), weights)


def test_a_full_vocabulary_draft_shares_the_target_head():
    config = _config()
    config.draft_vocab_size = VOCAB
    weights = _weights()
    weights["markov_head.markov_w2.weight"] = torch.empty(VOCAB, RANK, device="meta")
    del weights["lm_head.weight"]
    del weights["d2t"]
    validate_dspark_weights(config, weights)


@pytest.mark.parametrize(
    "name,value",
    [
        ("target_hidden_size", 4096),
        ("attention_bias", True),
        ("head_dim", None),
        ("markov_rank", None),
        ("eagle_aux_hidden_state_layer_ids", None),
    ],
)
def test_unpublished_layout_has_a_clear_scope_error(name, value):
    config = _config()
    setattr(config, name, value)
    with pytest.raises(ValueError, match="Only published Qwen3 DSpark checkpoint layouts"):
        validate_dspark_weights(config, _weights())
