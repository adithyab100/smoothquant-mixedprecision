import torch
from torch import nn
from functools import partial
from tqdm import tqdm


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class W4A4Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        importance=None,
        salient_prop=None,
        quant_bits=4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=quant_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=quant_bits)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

        self.salient_indices = None

        if importance is not None and salient_prop is not None:
            self.salient_indices = torch.topk(importance, int(salient_prop * importance.size(0)))[1]

    def to(self, *args, **kwargs):
        super(W4A4Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x.view(-1, x.shape[-1])
        original_x = x.clone()
        
        if self.salient_indices is not None:
            # Create a mask for non-salient indices
            non_salient_mask = torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
            non_salient_mask[self.salient_indices] = False
            
            # Only quantize non-salient features
            q_x = x.clone()
            non_salient_x = x[:, non_salient_mask]
            if non_salient_x.numel() > 0:  # Only quantize if there are non-salient features
                quantized_non_salient = self.act_quant(non_salient_x)
                q_x[:, non_salient_mask] = quantized_non_salient
        else:
            # If no salient indices specified, quantize everything
            q_x = self.act_quant(x)

        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        
        # Only apply output quantization to non-salient features if needed
        if self.output_quant_name != "None" and self.salient_indices is not None:
            q_y = y.clone()
            non_salient_y = y[:, non_salient_mask] if non_salient_mask.any() else None
            if non_salient_y is not None and non_salient_y.numel() > 0:
                quantized_non_salient_y = self.output_quant(non_salient_y)
                q_y[:, non_salient_mask] = quantized_non_salient_y
        else:
            q_y = self.output_quant(y)

        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False, importance=None, salient_prop=None, quant_bits=4
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W4A4Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            importance=importance,
            salient_prop=salient_prop,
            quant_bits=quant_bits,
        )
        outlier_weights = module.weight.data[:, new_module.salient_indices].clone()
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=quant_bits
            )  # use 4-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=quant_bits
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")

        outlier_weights = outlier_weights.to(new_module.weight.data.dtype).to(new_module.weight.data.device)
        if new_module.salient_indices is not None:
            new_module.weight.data[:, new_module.salient_indices] = outlier_weights

        # assert torch.equal(starting_weights.to(new_module.weight.data.device), new_module.weight.data), "MEWING"
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W4A4Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, salient_indices={self.salient_indices})"


def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True, input_feat=None, salient_prop=None, quant_bits=4
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in tqdm(model.model.named_modules(), desc="Quantizing OPT model"):
        # importance = None

        if isinstance(m, OPTDecoderLayer):
            importance = sum(input_feat["model." + name + ".fc1"]).float()
            m.fc1 = W4A4Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant, importance=importance, salient_prop=salient_prop, quant_bits=quant_bits
            )
            importance = sum(input_feat["model." + name + ".fc2"]).float()
            m.fc2 = W4A4Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant, importance=importance, salient_prop=salient_prop, quant_bits=quant_bits
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            importance = sum(input_feat["model." + name + ".q_proj"]).float()
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits
            )
            importance = sum(input_feat["model." + name + ".k_proj"]).float()
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits
            )
            importance = sum(input_feat["model." + name + ".v_proj"]).float()
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits
            )
            importance = sum(input_feat["model." + name + ".out_proj"]).float()
            m.out_proj = W4A4Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant, importance=importance, salient_prop=salient_prop, quant_bits=quant_bits
            )
    return model


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, quant_bits=4
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W4A4Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
            m.up_proj = W4A4Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
            m.down_proj = W4A4Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits
            )
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits
            )
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits
            )
            m.o_proj = W4A4Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, quant_bits=4
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W4A4Linear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
            m.w2 = W4A4Linear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
            m.w3 = W4A4Linear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits
            )
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits
            )
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits
            )
            m.o_proj = W4A4Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W4A4Linear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, quant_bits=4
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W4A4Linear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
            m.dense_4h_to_h = W4A4Linear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W4A4Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits
            )
            m.dense = W4A4Linear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, quant_bits=4
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            quant_bits=quant_bits,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            quant_bits=quant_bits,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            quant_bits=quant_bits,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            quant_bits=quant_bits,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
