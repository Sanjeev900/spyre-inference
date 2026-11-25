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

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""DSpark draft model: upstream's Qwen3 DSpark on the Spyre context path."""

from contextlib import contextmanager

import torch
import torch.nn as nn
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)

from spyre_inference.models.qwen3_dflash import (
    SpyreDFlashQwen3ForCausalLM,
    SpyreDFlashQwen3Model,
)
from spyre_inference.v1.spec_decode.checkpoint import validate_dspark_weights


class SpyreDSparkMarkovHead(DSparkMarkovHead):
    """Host-resident Markov head; its tables are read one row per draft step.

    markov_w2 deviates from upstream's replicated ParallelLMHead: a plain
    nn.Linear with the same [draft_vocab, rank] weight, so the checkpoint loads
    unchanged and the head never places a device copy.
    """

    markov_w1: nn.Embedding
    markov_w2: nn.Linear

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
        quant_config=None,
        retain_weight_for_gather: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)

    def prepare_for_spyre(self) -> None:
        # One fp32 host copy; the per-step [B, rank] embedding is cast, not the tables.
        self._w2_fp32 = self.markov_w2.weight.detach().float()

    def bias(self, markov_embed: torch.Tensor, logits_processor=None) -> torch.Tensor:
        # Host-side transition bias. logit_scale 1 is checked at config time, so
        # the processor's scale is a no-op and is not applied.
        return markov_embed.float() @ self._w2_fp32.t()


class SpyreQwen3DSparkModel(SpyreDFlashQwen3Model, Qwen3DSparkModel):
    markov_head: SpyreDSparkMarkovHead


class SpyreQwen3DSparkForCausalLM(SpyreDFlashQwen3ForCausalLM, Qwen3DSparkForCausalLM):
    """Upstream's DSpark model and loader, with the Markov loop on the host."""

    model: SpyreQwen3DSparkModel  # ty: ignore[incompatible-mutable-override]

    def __init__(self, *, vllm_config, prefix=""):
        with _spyre_dspark_classes():
            Qwen3DSparkForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        weights = dict(weights)
        validate_dspark_weights(self.config, weights)
        Qwen3DSparkForCausalLM.load_weights(self, weights.items())

    def prepare_for_spyre(self) -> None:
        super().prepare_for_spyre()
        self.model.markov_head.prepare_for_spyre()

    def propose_block(self, hidden_states, anchor_ids):
        # Upstream's _sample_sequential over host tensors: a per-step Markov bias
        # on the base logits, argmax in draft space, then the d2t map to target
        # ids. prev always holds target ids; markov_w1 is indexed by target ids.
        batch = anchor_ids.numel()
        block = hidden_states.shape[0] // batch
        base = self.compute_device_logits(hidden_states).view(batch, block, -1)
        host_base = self._copy_logits_to_cpu(base)[:, 1:, : self.config.draft_vocab_size]
        head = self.model.markov_head
        d2t = self.draft_id_to_target_id
        drafts = torch.empty(batch, host_base.shape[1], dtype=torch.int64)
        prev = anchor_ids
        for step in range(host_base.shape[1]):
            draft_id = (host_base[:, step] + head.bias(head.embed(prev))).argmax(-1)
            token = draft_id if d2t is None else draft_id + d2t[draft_id]
            drafts[:, step] = token
            prev = token
        return drafts.tolist()


@contextmanager
def _spyre_dspark_classes():
    # Upstream constructs both by module-global name, with no class hook.
    from vllm.model_executor.models import qwen3_dspark

    saved = qwen3_dspark.Qwen3DSparkModel, qwen3_dspark.DSparkMarkovHead
    qwen3_dspark.Qwen3DSparkModel = SpyreQwen3DSparkModel  # ty: ignore[invalid-assignment]
    qwen3_dspark.DSparkMarkovHead = SpyreDSparkMarkovHead  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        qwen3_dspark.Qwen3DSparkModel, qwen3_dspark.DSparkMarkovHead = saved
