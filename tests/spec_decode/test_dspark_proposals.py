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
    base = logits.clone()
    got = drafter.propose_block(logits, anchors)
    # Row 0 of each request is the bonus token; its logits must not matter.
    logits[0, 0] += 100.0
    logits[STEPS + 1, 0] += 100.0
    assert drafter.propose_block(logits, anchors) == got
    # Every proposal is a mapped target id, not a raw draft id.
    targets = (torch.arange(DRAFT_VOCAB) + drafter.draft_id_to_target_id).tolist()
    for request in got:
        assert all(token in targets for token in request)
