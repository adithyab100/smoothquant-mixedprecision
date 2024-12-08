import torch
import torch.nn as nn


def get_model_size(model: nn.Module, data_width=16, salient_prop = 0, group_size=-1):
    data_width_non_salient = data_width
    data_width_salient = 16
    if group_size != -1:
        data_width_non_salient += (16 + 4) / group_size
        data_width_salient += (16 + 4) / group_size

    avg_data_width = data_width_non_salient * (1 - salient_prop) + data_width_salient * salient_prop
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * avg_data_width