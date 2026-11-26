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

"""MRV2: DSpark speculator (vllm/v1/worker/gpu/spec_decode/dspark/speculator.py).

Upstream's DSparkSpeculator over the Spyre DFlash overrides. The class lives in
its own module so the dspark import chain (vllm.models.deepseek_v4, via
dspark.utils) stays behind the dispatch table's lazy builder.
"""

from contextlib import contextmanager

import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.worker.spyre.spec_decode import SpyreDFlashSpeculator


class SpyreDSparkSpeculator(SpyreDFlashSpeculator, DSparkSpeculator):
    """The Markov loop runs in the model's propose_block; the rest is DFlash's."""

    keep_on_host = ("model.markov_head.", "draft_id_to_target_id")

    def load_draft_model(self, target_model, target_attn_layer_names):
        with _host_side_sharing_comparison():
            return super().load_draft_model(target_model, target_attn_layer_names)


@contextmanager
def _host_side_sharing_comparison():
    """load_dspark_model compares the draft's fresh host embedding with the
    target's, which our loader already placed on Spyre; compare on the host, as
    aten::equal has no spyre kernel and torch.equal raises across devices.
    """
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    upstream = eagle_utils._should_share

    def host_side(eagle, flag, draft, target):
        if not getattr(eagle, flag, False):
            return True
        if draft is None or target is None:
            return False
        return torch.equal(draft.weight, convert(target.weight, device="cpu"))

    eagle_utils._should_share = host_side  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        eagle_utils._should_share = upstream
