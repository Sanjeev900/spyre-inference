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

"""Spyre adaptation of vLLM's Transformers backend.

Upstream's fusers replace HF's linear/norm/GLU modules with vLLM layers, which the Spyre
OOT registrations pick up on their own. Two things are left to HF's module code:

* RoPE — there is no RoPE fuser, so HF's ``rotary_emb`` survives and derives cos/sin
  inside the forward from int64 ``position_ids``, a cast torch-spyre cannot lower.
  Replaced here with a precomputed rotation cache and a matmul-only rotation.
* Models shipping both ``config.json`` and ``params.json`` parse into a bare
  ``PretrainedConfig``, which HF cannot build a model from.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from vllm.logger import init_logger
from vllm.model_executor.models.transformers import (
    TransformersEmbeddingModel,
    TransformersForCausalLM,
    TransformersForSequenceClassification,
)

from spyre_inference.custom_ops.head_pad import original_head_dim

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _build_rotation_cache(
    inv_freq: torch.Tensor,
    scaling: float,
    max_position: int,
    padded_head_dim: int | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[max_position, 2, 2, head_dim // 2]`` rotation matrices ``[[cos, -sin], [sin, cos]]``.

    Working from ``inv_freq``/``attention_scaling`` inherits whatever rope scaling the
    module being replaced had baked into them. *padded_head_dim* extends the cache with
    identity blocks, so a Q/K padded up to it passes its trailing dimensions through.
    """
    rope_half = inv_freq.shape[0]
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
    cos, sin = torch.cos(freqs) * scaling, torch.sin(freqs) * scaling
    rot = torch.stack([cos, -sin, sin, cos], dim=1).view(max_position, 2, 2, rope_half)

    if padded_head_dim is not None and padded_head_dim // 2 > rope_half:
        identity = torch.zeros(max_position, 2, 2, padded_head_dim // 2 - rope_half)
        identity[:, 0, 0, :] = 1.0
        identity[:, 1, 1, :] = 1.0
        rot = torch.cat([rot, identity], dim=-1)

    return rot.to(dtype)


def _apply_rope_matmul(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, L, D]`` by ``rot`` ``[B, L, 2, 2, D // 2]``.

    Multiply-and-reduce rather than HF's ``rotate_half`` cat: Spyre cannot restickify the
    halves that slicing a stick-aligned head_dim produces.
    """
    b, h, seq, head_dim = x.shape
    pairs = x.transpose(1, 2).reshape(b, seq, h, 2, head_dim // 2)
    out = rot.unsqueeze(2).mul(pairs.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


def _rope_frequencies(original: nn.Module) -> dict[str | None, tuple[torch.Tensor, float]]:
    """``{layer_type: (inv_freq, attention_scaling)}`` for the rope module being replaced.

    Models mixing global and sliding-window attention (Gemma 3, Olmo 3, ...) register one
    ``{layer_type}_inv_freq`` buffer per type and select between them on a third
    ``layer_type`` argument to ``forward``; a single rope is keyed under ``None``, its
    default.
    """
    layer_types = getattr(original, "layer_types", None)
    if not layer_types:
        scaling = float(getattr(original, "attention_scaling", 1.0))
        return {None: (original.get_buffer("inv_freq"), scaling)}

    freqs = {}
    for layer_type in layer_types:
        try:
            inv_freq = original.get_buffer(f"{layer_type}_inv_freq")
        except AttributeError:
            continue  # a layer type that does not rotate registers no buffer
        scaling = float(getattr(original, f"{layer_type}_attention_scaling", 1.0))
        freqs[layer_type] = (inv_freq, scaling)
    return freqs


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for an HF rotary embedding, returning ``(rot, None)`` in place of
    ``(cos, sin)``; the patched ``apply_rotary_pos_emb`` ignores the second element.

    The cache covers ``max_position`` up front rather than growing on demand: sizing it
    from the batch's positions needs an ``.item()``, so a host sync per step and a
    data-dependent guard ``torch.compile`` cannot trace.
    """

    def __init__(
        self,
        original: nn.Module,
        max_position: int,
        padded_head_dim: int | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        self._cpu_caches = {
            layer_type: _build_rotation_cache(
                inv_freq.to("cpu", torch.float32),
                scaling,
                max_position,
                padded_head_dim,
                dtype,
            )
            for layer_type, (inv_freq, scaling) in _rope_frequencies(original).items()
        }
        self._caches = self._cpu_caches

    def _apply(self, fn, recurse=True):
        # Prime the device caches when the model moves to Spyre, i.e. before compile, so only
        # the index_select is traced. They are not buffers (they are built after weight
        # loading) and there are no children, so super() has nothing to do.
        self._caches = {k: fn(v) for k, v in self._cpu_caches.items()}
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None):
        cache = self._caches[layer_type]
        rot = cache.index_select(0, position_ids.flatten())
        return rot.view(*position_ids.shape, *cache.shape[1:]), None


def _spyre_apply_rotary(q, k, rot, *args, **kwargs):
    """Rotate Q and K by the matrices ``_SpyreRotaryEmbedding`` returned."""
    return _apply_rope_matmul(q, rot), _apply_rope_matmul(k, rot)


def _rope_dispatch(original: Callable) -> Callable:
    """``apply_rotary_pos_emb`` replacement that hands stock HF calls to *original*.

    The patch lands on a modeling module in ``sys.modules`` and is never removed, so an HF
    model built later in the process has to keep working. Spyre's calls are the ones whose
    ``sin`` is the ``None`` standing in for ``_SpyreRotaryEmbedding``'s second return.
    """

    @functools.wraps(original)
    def apply_rotary_pos_emb(q, k, cos, sin=None, *args, **kwargs):
        if sin is None:
            return _spyre_apply_rotary(q, k, cos)
        return original(q, k, cos, sin, *args, **kwargs)

    apply_rotary_pos_emb._spyre_patched = True
    return apply_rotary_pos_emb


def _patch_xlm_roberta_gather(model: nn.Module) -> None:
    """Swap ``XLMRobertaEmbeddings`` for a subclass that skips the ``gather`` path on token segment IDs.

    HF's standard forward has two branches for ``token_type_ids``:
    * buffer present  → ``torch.gather`` (aten::gather, not supported on Spyre layout remapper)
    * buffer absent   → ``torch.zeros(..., dtype=torch.long)`` (triggers int64→int32 downcast and
                         subsequent layout/ReStickifyOpHBM compile crash on Spyre)

    Neither branch compiles on Spyre. Since XLM-RoBERTa only has one token type, all IDs are 0
    and token_type_embeddings always returns weight[0] broadcasted.
    The subclass overrides ``forward`` to directly slice and expand weight[0], bypassing the
    integer indexing lookup entirely, ensuring only float16/float32 operations exist.

    ``module.__class__`` is rewritten in-place before compilation so all state is preserved and
    Dynamo traces the clean, tensor-only replacement forward method seamlessly.
    """
    try:
        from transformers.models.xlm_roberta.modeling_xlm_roberta import (
            XLMRobertaEmbeddings,
        )
    except ImportError:
        return

    class _XLMRobertaEmbeddingsSpyre(XLMRobertaEmbeddings):
        def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                    inputs_embeds=None, past_key_values_length=0):
            if token_type_ids is None:
                # Retrieve the activation shape and device details from word embeddings
                if input_ids is not None:
                    batch_size, seq_length = input_ids.shape
                    dev = input_ids.device
                else:
                    batch_size, seq_length = inputs_embeds.shape[:2]
                    dev = inputs_embeds.device
                
                # Directly slice weight[0] and expand to match activation dimensions
                # weight[0] is already in the module's correct device and float dtype
                token_type_embeddings = (
                    self.token_type_embeddings.weight[0]
                    .view(1, 1, -1)
                    .expand(batch_size, seq_length, -1)
                )

                # Recreate the rest of the forward pass manually, bypassing the lookup
                if position_ids is None:
                    if input_ids is not None:
                        position_ids = self.create_position_ids_from_input_ids(
                            input_ids, self.padding_idx, past_key_values_length
                        )
                    else:
                        position_ids = self.create_position_ids_from_inputs_embeds(
                            inputs_embeds, self.padding_idx
                        )
                if inputs_embeds is None:
                    inputs_embeds = self.word_embeddings(input_ids)
                
                embeddings = inputs_embeds + token_type_embeddings
                embeddings = embeddings + self.position_embeddings(position_ids)
                embeddings = self.LayerNorm(embeddings)
                return self.dropout(embeddings)
            
            return super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

    for _, module in model.named_modules():
        if not isinstance(module, XLMRobertaEmbeddings):
            continue
        if type(module) is _XLMRobertaEmbeddingsSpyre:
            continue  # already patched

        module.__class__ = _XLMRobertaEmbeddingsSpyre
        logger.debug("Replaced XLMRobertaEmbeddings with gather-free subclass")


def _rope_at_original_head_dim(cfg, rope: nn.Module, orig_head_dim: int) -> nn.Module:
    """Rebuild *rope* at the pre-pad head_dim.

    HF derived ``inv_freq`` from the widened ``config.head_dim``, giving one frequency
    per padded pair instead of per real pair.
    """
    padded = cfg.head_dim
    cfg.head_dim = orig_head_dim
    try:
        return type(rope)(config=cfg)
    finally:
        cfg.head_dim = padded


class SpyreTransformersForCausalLM(TransformersForCausalLM):
    """Transformers backend with the Spyre RoPE replacement wired in."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        self._max_position = vllm_config.model_config.max_model_len
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("SpyreTransformersForCausalLM ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        result = super().load_weights(weights)
        self._patch_rope()
        return result

    @staticmethod
    def _fix_generic_config(vllm_config: VllmConfig) -> None:
        """Re-resolve the bare PretrainedConfig that vLLM's Mistral parser produces for
        repos shipping both config.json and params.json, which AutoModel.from_config
        rejects, and force HF-format weight loading. ``--config-format hf`` skips it."""
        hf_config = vllm_config.model_config.hf_config
        if type(hf_config) is not PretrainedConfig:
            return

        model_id = vllm_config.model_config.hf_config_path or vllm_config.model_config.model
        try:
            resolved = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.revision,
            )
        except Exception:
            logger.warning("AutoConfig re-resolve failed for %s", model_id, exc_info=True)
            return

        skip = {"model_type", "_name_or_path", "transformers_version", "auto_map", "architectures"}
        for key, val in hf_config.to_dict().items():
            if key not in skip and val is not None:
                setattr(resolved, key, val)

        vllm_config.model_config.hf_config = resolved
        vllm_config.model_config.hf_text_config = resolved.get_text_config()
        if vllm_config.load_config.load_format in ("auto", "mistral"):
            vllm_config.load_config.load_format = "hf"
        logger.debug(
            "Re-resolved config: %s (model_type=%s), load_format=hf",
            type(resolved).__name__,
            resolved.model_type,
        )

    def _patch_rope(self):
        """Swap HF's rotary embedding and ``apply_rotary_pos_emb`` for the Spyre ones.

        Partial rotary dimensions (e.g. Phi-3) are unsupported — the cache would cover
        only the rotated dims — but reach a shape mismatch here rather than a check:
        ``_maybe_pad_head_dim`` already rejects them whenever padding is needed.
        """
        # The text backbone holding rotary_emb; multimodal models nest it one level
        # deeper, at model.model.language_model, and carry the rope config on its own
        # config rather than the top-level one.
        inner = self.model.model if hasattr(self.model, "model") else self.model
        backbone = cast(nn.Module, getattr(inner, "language_model", inner))
        cfg = getattr(backbone, "config", self.model.config)

        # head_dim is already stick-aligned (the platform pads it, and the weight pass
        # pads Q/K interleaved to match), so the rotation only needs the pre-pad
        # frequencies identity-padded back out to the widened width.
        rope_source = backbone.get_submodule("rotary_emb")
        orig_head_dim = original_head_dim(cfg)
        padded_head_dim = None
        if orig_head_dim is not None:
            padded_head_dim = cfg.head_dim
            rope_source = _rope_at_original_head_dim(cfg, rope_source, orig_head_dim)

        spyre_rope = _SpyreRotaryEmbedding(
            rope_source,
            self._max_position,
            padded_head_dim,
            next(self.model.parameters()).dtype,
        )
        backbone.rotary_emb = spyre_rope

        patched_mods: set[int] = set()
        for name, module in self.model.named_modules():
            if module is spyre_rope:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding"):
                parent_name, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, attr, spyre_rope)
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = spyre_rope

            # apply_rotary_pos_emb is a module-level function in HF modeling files, so it
            # is patched once per modeling module rather than per layer.
            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            existing = getattr(mod, "apply_rotary_pos_emb", None)
            if existing is None or getattr(existing, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _rope_dispatch(existing)
            patched_mods.add(id(mod))


# using_transformers_backend() compares _ModelInfo.architecture, which is model_cls.__name__,
# against "TransformersForCausalLM", so the subclass has to keep answering to that name.
SpyreTransformersForCausalLM.__name__ = "TransformersForCausalLM"


class SpyreTransformersEmbeddingModel(TransformersEmbeddingModel):
    """Transformers backend for pooling/embed models with the Spyre RoPE replacement.

    Encoder models that use absolute position embeddings (BERT, RoBERTa, XLM-RoBERTa)
    have no ``rotary_emb`` on their backbone; ``_patch_rope`` is a no-op for them.
    Models that do use RoPE (e.g. NomicBERT / Granite-125m) go through the same
    matmul-based rotation as the decoder adapter.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        SpyreTransformersForCausalLM._fix_generic_config(vllm_config)
        self._max_position = vllm_config.model_config.max_model_len
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # RoBERTa/XLM-RoBERTa checkpoints save position_ids as a persistent buffer;
        # vLLM's loader rejects it as unexpected because the module registers it as
        # non-persistent. Safe to ignore — it is recreated at runtime.
        self.ignore_unexpected_suffixes.append("position_ids")
        logger.debug("SpyreTransformersEmbeddingModel ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        result = super().load_weights(weights)
        _patch_xlm_roberta_gather(self.model)
        self._patch_rope()
        return result

    def _patch_rope(self):
        """Swap HF's rotary embedding for the Spyre matmul-based one, if present.

        Most encoder architectures (BERT, RoBERTa, XLM-RoBERTa) use absolute position
        embeddings and have no ``rotary_emb`` on the model — for those this is a no-op.
        NomicBERT / Granite-125m do use RoPE and go through the full patch path.
        """
        if not hasattr(self.model, "rotary_emb"):
            return

        cfg = getattr(self.model, "config", self.model.config)

        rope_source = self.model.get_submodule("rotary_emb")
        orig_head_dim = original_head_dim(cfg)
        padded_head_dim = None
        if orig_head_dim is not None:
            padded_head_dim = cfg.head_dim
            rope_source = _rope_at_original_head_dim(cfg, rope_source, orig_head_dim)

        spyre_rope = _SpyreRotaryEmbedding(
            rope_source,
            self._max_position,
            padded_head_dim,
            next(self.model.parameters()).dtype,
        )
        self.model.rotary_emb = spyre_rope

        patched_mods: set[int] = set()
        for name, module in self.model.named_modules():
            if module is spyre_rope:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding"):
                parent_name, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, attr, spyre_rope)
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = spyre_rope

            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            existing = getattr(mod, "apply_rotary_pos_emb", None)
            if existing is None or getattr(existing, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _rope_dispatch(existing)
            patched_mods.add(id(mod))


# Same aliasing requirement as SpyreTransformersForCausalLM.
SpyreTransformersEmbeddingModel.__name__ = "TransformersEmbeddingModel"


class SpyreTransformersForSequenceClassification(TransformersForSequenceClassification):
    """Transformers backend for pooling/classify (reranker) models on Spyre."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        SpyreTransformersForCausalLM._fix_generic_config(vllm_config)
        self._max_position = vllm_config.model_config.max_model_len
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.ignore_unexpected_suffixes.append("position_ids")

        # Clean Fix: Alias the pre_classifier if the underlying model has one
        # to ensure vLLM's SequenceClassification weight loader finds it.
        if hasattr(self.model, "pre_classifier") and not hasattr(self, "pre_classifier"):
            self.pre_classifier = self.model.pre_classifier

        logger.debug(
            "SpyreTransformersForSequenceClassification ready: %s",
            type(self.model).__name__,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        result = super().load_weights(weights)
        _patch_xlm_roberta_gather(self.model)
        self._patch_rope()
        return result

    def _patch_rope(self):
        SpyreTransformersEmbeddingModel._patch_rope(self)


# Same aliasing requirement as SpyreTransformersForCausalLM.
SpyreTransformersForSequenceClassification.__name__ = "TransformersForSequenceClassification"
