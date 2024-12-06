import torch
from torch import nn
from functools import partial
from tqdm import tqdm
from scipy import stats
import numpy as np


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
def quantize_weight_per_group_absmax(w, n_bits, group_size=128):
    # w: (out_features, in_features)
    w_shape = w.shape
    num_groups = (w_shape[1] + group_size - 1) // group_size
    
    # Pad if needed
    if w_shape[1] % group_size != 0:
        pad_size = group_size - (w_shape[1] % group_size)
        w = torch.nn.functional.pad(w, (0, pad_size))
    
    # Reshape to group dimension
    w_grouped = w.view(w_shape[0], num_groups, group_size)
    
    # Compute scales per group
    scales = w_grouped.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    
    # Quantize each group
    w_grouped.div_(scales).round_().mul_(scales)
    
    # Reshape back and remove padding
    w = w_grouped.view(w_shape[0], -1)[:, :w_shape[1]]
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_group_absmax(t, n_bits, group_size=128):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    num_groups = (t.shape[-1] + group_size - 1) // group_size
    
    # Pad if needed
    if t.shape[-1] % group_size != 0:
        pad_size = group_size - (t.shape[-1] % group_size)
        t = torch.nn.functional.pad(t, (0, pad_size))
    
    # Reshape to group dimension
    t_grouped = t.view(t.shape[0], num_groups, group_size)
    
    # Compute scales per group
    scales = t_grouped.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    
    # Quantize each group
    t_grouped.div_(scales).round_().mul_(scales)
    
    # Reshape back and remove padding
    t = t_grouped.view(t.shape[0], -1)[:, :t_shape[-1]]
    return t.view(t_shape)


def compute_adaptive_salient_prop(importance, min_prop=0, max_prop=0):
    """
    Compute adaptive salient proportion based on importance skewness.
    - High skewness (many low importance values) -> lower proportion
    - Low skewness (more uniform importance) -> higher proportion
    """
    if importance is None:
        return None
        
    # Calculate skewness of importance values
    importance_np = importance.cpu().numpy()
    skewness = stats.skew(importance_np)
    
    # Normalize skewness to [0, 1] range using sigmoid
    normalized_skew = 1 / (1 + np.exp(-skewness))
    
    # Inverse relationship: high skew -> low proportion
    prop = max_prop - (max_prop - min_prop) * normalized_skew
    
    # Ensure proportion is within bounds
    return float(np.clip(prop, min_prop, max_prop))


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
        group_size=128,
        min_prop=0,
        max_prop=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

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
        elif act_quant == "per_group":
            self.act_quant_name = "per_group"
            self.act_quant = partial(quantize_activation_per_group_absmax, n_bits=quant_bits, group_size=group_size)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

        self.salient_indices = None
        if importance is not None:
            adaptive_prop = compute_adaptive_salient_prop(importance, min_prop=min_prop, max_prop=max_prop)
            if adaptive_prop is not None:
                self.salient_indices = torch.topk(importance, int(adaptive_prop * importance.size(0)))[1]

    def to(self, *args, **kwargs):
        super(W4A4Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 3:  # [batch_size, seq_len, hidden_dim]
            x = x.reshape(-1, x_shape[-1])  # Combine batch and seq_len dimensions
        elif len(x_shape) == 2:  # [batch_size, hidden_dim]
            pass  # Already in correct shape
        else:
            raise ValueError(f"Unsupported input shape: {x_shape}")
            
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

        # Restore original shape
        if len(x_shape) == 3:
            return q_y.view(x_shape[0], x_shape[1], -1)  # [batch_size, seq_len, hidden_dim]
        else:
            return q_y  # [batch_size, hidden_dim]

    @staticmethod
    def from_float(
        module, 
        weight_quant="per_channel", 
        act_quant="per_token", 
        quantize_output=False, 
        importance=None, 
        salient_prop=None, 
        quant_bits=4, 
        group_size=128,
        min_prop=0,
        max_prop=0,
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
            group_size=group_size,
            min_prop=min_prop,
            max_prop=max_prop,
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
        elif weight_quant == "per_group":
            new_module.weight = quantize_weight_per_group_absmax(
                module.weight, n_bits=quant_bits, group_size=group_size
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
    model, 
    weight_quant="per_tensor", 
    act_quant="per_tensor", 
    quantize_bmm_input=True, 
    input_feat=None, 
    salient_prop=None, 
    quant_bits=4, 
    group_size=128,
    min_prop=0,
    max_prop=0,
):
    max_prop = max(min_prop, max_prop)
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in tqdm(model.model.named_modules(), desc="Quantizing OPT model"):
        # importance = None

        if isinstance(m, OPTDecoderLayer):
            importance = sum(input_feat["model." + name + ".fc1"]).float()
            m.fc1 = W4A4Linear.from_float(
                m.fc1, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                importance=importance, 
                salient_prop=salient_prop, 
                quant_bits=quant_bits, 
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".fc2"]).float()
            m.fc2 = W4A4Linear.from_float(
                m.fc2, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                importance=importance, 
                salient_prop=salient_prop, 
                quant_bits=quant_bits, 
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
        elif isinstance(m, OPTAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            importance = sum(input_feat["model." + name + ".q_proj"]).float()
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits,
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".k_proj"]).float()
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits,
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".v_proj"]).float()
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits,
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".out_proj"]).float()
            m.out_proj = W4A4Linear.from_float(
                m.out_proj, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                importance=importance, 
                salient_prop=salient_prop, 
                quant_bits=quant_bits, 
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
    return model


def quantize_llama_like(
    model, 
    weight_quant="per_channel", 
    act_quant="per_token", 
    quantize_bmm_input=False, 
    input_feat=None, 
    salient_prop=None, 
    quant_bits=4, 
    group_size=128,
    min_prop=0,
    max_prop=0,
):
    max_prop = max(min_prop, max_prop)
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
            importance = sum(input_feat["model." + name + ".gate_proj"]).float() if input_feat else None
            m.gate_proj = W4A4Linear.from_float(
                m.gate_proj, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                importance=importance, 
                salient_prop=salient_prop, 
                quant_bits=quant_bits, 
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".up_proj"]).float() if input_feat else None
            m.up_proj = W4A4Linear.from_float(
                m.up_proj, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                importance=importance, 
                salient_prop=salient_prop, 
                quant_bits=quant_bits, 
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".down_proj"]).float() if input_feat else None
            m.down_proj = W4A4Linear.from_float(
                m.down_proj, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                importance=importance, 
                salient_prop=salient_prop, 
                quant_bits=quant_bits, 
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            importance = sum(input_feat["model." + name + ".q_proj"]).float() if input_feat else None
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits,
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".k_proj"]).float() if input_feat else None
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits,
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".v_proj"]).float() if input_feat else None
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
                quant_bits=quant_bits,
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            importance = sum(input_feat["model." + name + ".o_proj"]).float() if input_feat else None
            m.o_proj = W4A4Linear.from_float(
                m.o_proj, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                importance=importance, 
                salient_prop=salient_prop, 
                quant_bits=quant_bits, 
                group_size=group_size,
                min_prop=min_prop,
                max_prop=max_prop,
            )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, quant_bits=4, group_size=128
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W4A4Linear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits, group_size=group_size
            )
            m.w2 = W4A4Linear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits, group_size=group_size
            )
            m.w3 = W4A4Linear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits, group_size=group_size
            )
        elif isinstance(m, MixtralAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits,
                group_size=group_size,
            )
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits,
                group_size=group_size,
            )
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits,
                group_size=group_size,
            )
            m.o_proj = W4A4Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits, group_size=group_size
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W4A4Linear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits, group_size=group_size
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, quant_bits=4, group_size=128
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W4A4Linear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits, group_size=group_size
            )
            m.dense_4h_to_h = W4A4Linear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits, group_size=group_size
            )
        elif isinstance(m, FalconAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W4A4Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quant_bits=quant_bits,
                group_size=group_size,
            )
            m.dense = W4A4Linear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant, quant_bits=quant_bits, group_size=group_size
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, quant_bits=4, group_size=128
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
            group_size=group_size,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            quant_bits=quant_bits,
            group_size=group_size,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            quant_bits=quant_bits,
            group_size=group_size,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            quant_bits=quant_bits,
            group_size=group_size,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
