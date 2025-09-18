# =====================================================
# network/pipeline.py - 性能分析
# =====================================================
import torch
import numpy as np
from thop import profile, clever_format
from ptflops import get_model_complexity_info

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()


def cal_params_thop(model, tensor):
    flops, params = profile(model, inputs = (tensor, ))
    flops, params = clever_format([flops, params], '%.3f')
    return flops, params

def cal_params_ptflops(model, shape):
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, shape, as_strings=True, print_per_layer_stat=True, verbose=True)
    return flops, params
