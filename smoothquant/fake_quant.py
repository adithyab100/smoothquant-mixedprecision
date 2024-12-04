import torch
from torch import nn
from functools import partial

# Global quantization bits configuration
QUANT_BITS = 4


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=QUANT_BITS):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=QUANT_BITS):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=QUANT_BITS):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=QUANT_BITS):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_with_dual_scales(w, n_bits=QUANT_BITS, salient_indices=None):
    """Quantize weights using separate scales for salient and non-salient weights"""
    if salient_indices is None:
        return quantize_weight_per_channel_absmax(w, n_bits)
    
    print("\n=== Quantization Debug Info ===")
    print(f"Input weight stats: mean={w.mean():.6f}, std={w.std():.6f}, max={w.abs().max():.6f}")
    print(f"Number of salient indices: {len(salient_indices)}")
    
    # Create masks for salient and non-salient weights
    salient_mask = torch.zeros_like(w, dtype=torch.bool)
    salient_mask[:, salient_indices] = True
    
    # Original salient weights stats
    orig_salient = w[salient_mask].clone()
    print(f"Original salient weights: mean={orig_salient.mean():.6f}, std={orig_salient.std():.6f}, max={orig_salient.abs().max():.6f}")
    
    # Handle non-salient weights with normal quantization
    w_non_salient = w.clone()
    w_non_salient[:, salient_indices] = 0
    scales_non_salient = w_non_salient.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales_non_salient.clamp_(min=1e-5).div_(q_max)
    w_non_salient.div_(scales_non_salient).round_().mul_(scales_non_salient)
    
    print(f"Quantized non-salient stats: mean={w_non_salient[~salient_mask].mean():.6f}, std={w_non_salient[~salient_mask].std():.6f}")
    print(f"Non-salient scale mean: {scales_non_salient.mean():.6f}")
    
    # Handle salient weights - scale them to maintain relative importance
    w_salient = w.clone()
    w_salient[~salient_mask] = 0
    # Scale salient weights relative to quantized scale
    salient_max = w_salient.abs().max()
    relative_scale = scales_non_salient.mean() / salient_max if salient_max > 0 else 1.0
    w_salient.mul_(relative_scale)
    
    print(f"Salient max before scaling: {salient_max:.6f}")
    print(f"Relative scale factor: {relative_scale:.6f}")
    print(f"Scaled salient stats: mean={w_salient[salient_mask].mean():.6f}, std={w_salient[salient_mask].std():.6f}")
    
    # Combine both parts
    result = w_salient + w_non_salient
    print(f"Final combined stats: mean={result.mean():.6f}, std={result.std():.6f}")
    print(f"Final salient stats: mean={result[salient_mask].mean():.6f}, std={result[salient_mask].std():.6f}")
    print(f"Final non-salient stats: mean={result[~salient_mask].mean():.6f}, std={result[~salient_mask].std():.6f}")
    print("==============================\n")
    
    return result


@torch.no_grad()
def quantize_gradients_hook(grad, salient_indices=None, n_bits=QUANT_BITS):
    """
    A hook to quantize the gradients during backpropagation, applying quantization only to non-salient gradients.
    Salient gradients are preserved in full precision.
    """
    if salient_indices is not None:
        # Make a copy of the gradients to avoid modifying the original tensor in-place
        grad = grad.clone()

        # Set gradients of the salient indices to zero or preserve them
        grad[salient_indices] = grad[salient_indices]  # Preserve the gradients for salient indices

        # Quantize the non-salient gradients
        non_salient_indices = torch.setdiff1d(torch.arange(grad.size(0)), salient_indices)
        grad[non_salient_indices] = grad[non_salient_indices] / (2 ** (n_bits - 1) - 1)
        grad[non_salient_indices] = grad[non_salient_indices].round()

    return grad


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
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=QUANT_BITS)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=QUANT_BITS)
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
            # print("SKIBIDI")
            self.salient_indices = torch.topk(importance, int(salient_prop * importance.size(0)))[
                1
            ]
            # print(self.salient_indices)
            # raise NotImplementedError
            # print("FROM THE SCREEN TO THE RING TO PEN TO THE KING ", len(self.salient_indices), self.in_features, self.out_features)

    def to(self, *args, **kwargs):
        super(W4A4Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x.view(-1, x.shape[-1])
        x_salient = x[:, self.salient_indices].clone()
        q_x = self.act_quant(x)
        # preserve salient activations
        if self.salient_indices is not None:
            q_x[:, self.salient_indices] = x_salient

        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)

        # apply backward gradient quantization only for non-salient indices
        if self.training and self.salient_indices is not None:
            q_y.register_hook(
                partial(quantize_gradients_hook, salient_indices=self.salient_indices, n_bits=QUANT_BITS)
            )

        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False, importance=None, salient_prop=None
    ):
        assert isinstance(module, torch.nn.Linear)
        print(f"\n=== Processing Layer: in_features={module.in_features}, out_features={module.out_features} ===")
        
        new_module = W4A4Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            importance=importance,
            salient_prop=salient_prop,
        )
        
        if importance is not None and salient_prop is not None:
            print(f"Importance score range: {importance.min():.6f} to {importance.max():.6f}")
            print(f"Salient proportion: {salient_prop}")
            print(f"Number of salient indices: {len(new_module.salient_indices)}")
        
        # Quantize weights using dual scales if we have salient indices
        new_module.weight = quantize_with_dual_scales(
            module.weight, 
            n_bits=QUANT_BITS,
            salient_indices=new_module.salient_indices
        )
        
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W4A4Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, salient_indices={self.salient_indices})"


def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True, input_feat=None, salient_prop=None
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        # importance = None

        if isinstance(m, OPTDecoderLayer):
            importance = sum(input_feat["model." + name + ".fc1"]).float()
            m.fc1 = W4A4Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant, importance=importance, salient_prop=salient_prop
            )
            importance = sum(input_feat["model." + name + ".fc2"]).float()
            m.fc2 = W4A4Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant, importance=importance, salient_prop=salient_prop
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
            )
            importance = sum(input_feat["model." + name + ".k_proj"]).float()
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
            )
            importance = sum(input_feat["model." + name + ".v_proj"]).float()
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                importance=importance,
                salient_prop=salient_prop,
            )
            importance = sum(input_feat["model." + name + ".out_proj"]).float()
            m.out_proj = W4A4Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant, importance=importance, salient_prop=salient_prop
            )
            # print("OHIO")
    return model


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
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
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W4A4Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W4A4Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W4A4Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W4A4Linear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w2 = W4A4Linear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w3 = W4A4Linear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W4A4Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W4A4Linear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W4A4Linear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant
            )
            m.dense_4h_to_h = W4A4Linear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W4A4Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.dense = W4A4Linear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
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
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
