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

"""propose_block's Markov loop against a straight-line float32 reference."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spyre_inference.models.qwen3_dspark import (
    SpyreDSparkMarkovHead,
    SpyreQwen3DSparkForCausalLM,
)

VOCAB, DRAFT_VOCAB, RANK, STEPS, BATCH = 256, 64, 8, 3, 2


def _drafter(seed=0):
    torch.manual_seed(seed)
    head = SpyreDSparkMarkovHead(VOCAB, DRAFT_VOCAB, RANK, prefix="")
    nn.init.normal_(head.markov_w1.weight)
    nn.init.normal_(head.markov_w2.weight)
    head.prepare_for_spyre()
    drafter = SpyreQwen3DSparkForCausalLM.__new__(SpyreQwen3DSparkForCausalLM)
    drafter.model = SimpleNamespace(markov_head=head)
    drafter.config = SimpleNamespace(draft_vocab_size=DRAFT_VOCAB)
    # Draft id i maps to target id 3i: a non-trivial, strictly increasing map.
    drafter.draft_id_to_target_id = torch.arange(DRAFT_VOCAB) * 3
    # Stand in for the device logits path: the "hidden states" are the logits.
    drafter.compute_device_logits = lambda hidden: hidden
    drafter._copy_logits_to_cpu = lambda logits: logits.float().cpu()
    return drafter, head


def _reference(head, base, anchors, d2t):
    """Upstream's _sample_sequential, straight-line and independent of the head."""
    prev = anchors.clone()
    proposals = []
    for step in range(STEPS):
        embed = head.markov_w1(prev).float()
        bias = embed @ head.markov_w2.weight.detach().float().t()
        draft_id = (base[:, step] + bias).argmax(-1)
        token = draft_id + d2t[draft_id]
        proposals.append(token.tolist())
        prev = token
    return [list(request) for request in zip(*proposals)]


def test_the_markov_loop_matches_the_reference():
    drafter, head = _drafter()
    torch.manual_seed(1)
    # One bonus row plus STEPS mask rows per request, as the fill-in layout builds.
    logits = torch.randn(BATCH * (STEPS + 1), DRAFT_VOCAB)
    anchors = torch.tensor([5, 200])
    got = drafter.propose_block(logits, anchors)
    want = _reference(
        head, logits.view(BATCH, STEPS + 1, -1)[:, 1:, :], anchors, drafter.draft_id_to_target_id
    )
    assert got == want


def test_the_anchor_row_is_dropped_and_prev_holds_target_ids():
    drafter, head = _drafter()
    torch.manual_seed(2)
    logits = torch.randn(BATCH * (STEPS + 1), DRAFT_VOCAB)
    anchors = torch.tensor([7, 199])
    got = drafter.propose_block(logits, anchors)
    # Row 0 of each request is the bonus token; its logits must not matter.
    logits[0, 0] += 100.0
    logits[STEPS + 1, 0] += 100.0
    assert drafter.propose_block(logits, anchors) == got
    # Every proposal is a mapped target id, not a raw draft id.
    targets = (torch.arange(DRAFT_VOCAB) + drafter.draft_id_to_target_id).tolist()
    for request in got:
        assert all(token in targets for token in request)


def test_the_sampler_branch_draws_real_rows_and_leaves_padding_greedy():
    drafter, _ = _drafter()
    torch.manual_seed(3)
    logits = torch.randn(BATCH * (STEPS + 1), DRAFT_VOCAB)
    anchors = torch.tensor([11, 199])
    greedy = drafter.propose_block(logits, anchors)
    calls = []

    def sampler(biased, step, num_reqs):
        calls.append((step, num_reqs))
        return torch.full((num_reqs,), 200, dtype=torch.int64)

    got = drafter.propose_block(logits, anchors, num_reqs=1, sampler=sampler)
    assert calls == [(s, 1) for s in range(STEPS)]
    # The real row carries the draw; the padded row keeps its greedy trajectory
    # (each row's Markov chain feeds only its own previous token).
    assert got[0] == [200] * STEPS
    assert got[1] == greedy[1]


def test_device_logits_gather_the_sharded_vocabulary():
    """Under TP the draft lm_head yields this rank's vocab shard; the full
    vocabulary returns through the logits processor's gather."""
    drafter, _ = _drafter()
    # Drop the fixture's stand-in (a __dict__ entry; nn.Module.__delattr__ needs
    # the full init) so the class's real method runs.
    drafter.__dict__.pop("compute_device_logits")
    gathered = []
    drafter.lm_head = SimpleNamespace(
        tp_size=4, quant_method=SimpleNamespace(apply=lambda head, hidden: hidden * 2)
    )
    drafter.logits_processor = SimpleNamespace(
        _gather_logits=lambda logits: gathered.append(logits) or logits + 1
    )
    hidden = torch.randn(3, 8)
    assert torch.equal(drafter.compute_device_logits(hidden), hidden * 2 + 1)
    assert len(gathered) == 1 and torch.equal(gathered[0], hidden * 2)

    drafter.lm_head.tp_size = 1
    assert torch.equal(drafter.compute_device_logits(hidden), hidden * 2)
    assert len(gathered) == 1


@pytest.fixture
def upstream_dspark_model(monkeypatch):
    """The real upstream construction chain under our class swap, with the
    backbone and the head-consuming vLLM layers stubbed to plain torch."""
    from vllm.model_executor.models import qwen3_dspark
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    from spyre_inference.models.qwen3_dspark import SpyreQwen3DSparkForCausalLM

    config = SimpleNamespace(
        vocab_size=73,
        hidden_size=128,
        head_dim=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        num_hidden_layers=0,
        dflash_config={"target_layer_ids": [1, 2]},
        eagle_aux_hidden_state_layer_ids=[1, 2],
        markov_rank=16,
        draft_vocab_size=37,
        enable_confidence_head=False,
        logit_scale=1.0,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config)),
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: 4, get_vocab_size=lambda: 73
        ),
    )

    def backbone_init(self, *, vllm_config, start_layer_id=0, prefix=""):
        torch.nn.Module.__init__(self)
        self.config = config
        self.quant_config = None
        self.fc = torch.nn.Linear(256, 128, bias=False)
        self.hidden_norm = torch.nn.RMSNorm(128)
        self.norm = torch.nn.RMSNorm(128)
        self.layers = torch.nn.ModuleList()

    monkeypatch.setattr(DFlashQwen3Model, "__init__", backbone_init)
    monkeypatch.setattr(
        qwen3_dspark,
        "ParallelLMHead",
        lambda draft_vocab, hidden, prefix="": torch.nn.Linear(hidden, draft_vocab, bias=False),
    )
    monkeypatch.setattr(qwen3_dspark, "LogitsProcessor", lambda *a, **k: SimpleNamespace(scale=1.0))
    return SpyreQwen3DSparkForCausalLM(vllm_config=vllm_config)


def test_upstream_dspark_construction_builds_the_spyre_classes(upstream_dspark_model):
    from vllm.model_executor.models import qwen3_dspark

    from spyre_inference.models.qwen3_dspark import (
        SpyreDSparkMarkovHead,
        SpyreQwen3DSparkModel,
    )

    model = upstream_dspark_model
    assert type(model.model) is SpyreQwen3DSparkModel
    head = model.model.markov_head
    assert type(head) is SpyreDSparkMarkovHead
    assert head.markov_w1.weight.shape == (73, 16)
    assert head.markov_w2.weight.shape == (37, 16)
    assert model.model.confidence_head is None
    assert model.draft_id_to_target_id.shape == (37,)
    assert qwen3_dspark.Qwen3DSparkModel is not SpyreQwen3DSparkModel, "class swap leaked"
    assert qwen3_dspark.DSparkMarkovHead is not SpyreDSparkMarkovHead, "class swap leaked"


def test_upstream_dspark_checkpoint_roundtrip(upstream_dspark_model):
    """load_weights: the Spyre checker gates raw checkpoint names; upstream's
    loader renames d2t, prefixes model., skips t2d, and lands every tensor."""
    model = upstream_dspark_model
    weights = {}
    for name, value in model.state_dict().items():
        if name == "draft_id_to_target_id":
            weights["d2t"] = torch.arange(value.shape[0]) * 2
        elif name.startswith("model."):
            weights[name.removeprefix("model.")] = torch.full_like(value, 0.25)
        else:
            weights[name] = torch.full_like(value, 0.25)
    weights["t2d"] = torch.zeros(73, dtype=torch.bool)
    weights["confidence_head.proj.weight"] = torch.zeros(1, 144)
    model.load_weights(weights.items())

    torch.testing.assert_close(model.draft_id_to_target_id, torch.arange(37) * 2, atol=0, rtol=0)
    torch.testing.assert_close(
        model.model.markov_head.markov_w1.weight, torch.full((73, 16), 0.25), atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.model.markov_head.markov_w2.weight, torch.full((37, 16), 0.25), atol=0, rtol=0
    )
    torch.testing.assert_close(model.lm_head.weight, torch.full((37, 128), 0.25), atol=0, rtol=0)
