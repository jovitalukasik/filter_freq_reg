from torch.nn.init import _calculate_correct_fan, calculate_gain
import torch.nn as nn
import torch
import torch.nn.functional as F

import warnings
import math
import numpy as np
import itertools
import efficientnet_pytorch


def replace_conv2d(module, new_cls, kernel_size = 3, dct_kernel_size=3, filter_cb=None,**kwargs):
    for name, child in module.named_children():
        if name == "basis_coef":
            continue
        if type(child) == torch.nn.Conv2d and (not filter_cb or (filter_cb and filter_cb(child))) and child.kernel_size==(kernel_size,kernel_size)  :
            new_module = new_cls(in_channels=child.in_channels, out_channels=child.out_channels, kernel_size=dct_kernel_size, 
                                 padding=child.padding, stride=child.stride, dilation=child.dilation, groups=child.groups, conv_kernel_size=child.kernel_size[0], **kwargs)
            setattr(module, name, new_module)
        elif  type(child) == efficientnet_pytorch.utils.Conv2dStaticSamePadding and (not filter_cb or (filter_cb and filter_cb(child))) and child.kernel_size==(kernel_size,kernel_size)  :
            new_module = new_cls(in_channels=child.in_channels, out_channels=child.out_channels, kernel_size=dct_kernel_size, 
                                 padding=child.padding, stride=child.stride, dilation=child.dilation, groups=child.groups, static_padding=child.static_padding, conv_kernel_size=child.kernel_size[0],**kwargs)
            setattr(module, name, new_module)

    for child in module.children():
        replace_conv2d(child, new_cls, kernel_size, dct_kernel_size, filter_cb, **kwargs)

def kaiming_uniform_receptive_(tensor, fan, a=0, mode='fan_in', nonlinearity='relu'):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor

    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

    
def kaiming_normal_receptive_(tensor,fan, a=0, mode='fan_in', nonlinearity='relu'):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor

    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)
    
    
def dct_2_basis(N=3):
    B = torch.empty((N, N, N, N))
    for k, l, m, n in itertools.product(range(N), repeat=4):
        B[k, l, m, n] = np.cos((np.pi / N) * (k) * (n + 0.5)) * np.cos(
            (np.pi / N) * (l) * (m + 0.5)
        )
    return B


def change_basis(X, Bt):
    return np.dot(X - X.mean(axis=0), Bt).astype(np.float32)


def normalize_basis(b, N):
    norm = np.linalg.norm(b.reshape(-1, N*N), axis=-1, ord=1)
    b = b / norm.reshape(N, N, 1, 1)
    return b


class Weight_Decomposition(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  padding=1, stride=1, dilation=1, groups=1, init_scale="kaiming_uniform", bias=0, static_padding=None, conv_kernel_size=3):
        super(Weight_Decomposition, self).__init__()
        assert init_scale in ["kaiming_uniform", "kaiming_normal", "kaiming_beta", "ones"]
        self.init_scale = init_scale
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = None
        self.kernel_size = conv_kernel_size
        self.static_padding = static_padding
        self.conv_kernel_size = conv_kernel_size  

        if kernel_size < conv_kernel_size:
            self.basis = torch.Tensor(normalize_basis(dct_2_basis(N=conv_kernel_size),conv_kernel_size))
            self.basis[:,kernel_size:conv_kernel_size] = 0
            self.basis[kernel_size:conv_kernel_size,:] = 0

        else:      
            self.basis = torch.Tensor(normalize_basis(dct_2_basis(N=kernel_size),kernel_size))
               
            
        if conv_kernel_size < kernel_size:
            self.basis = self.basis[:conv_kernel_size, :conv_kernel_size, :conv_kernel_size, :conv_kernel_size].contiguous()

        self.basis = torch.nn.Parameter(self.basis,requires_grad=False)

        self.coefficients = torch.nn.Parameter(torch.empty(in_channels * out_channels // groups, self.kernel_size * self.kernel_size, 1, 1),
                                               requires_grad=True)  # c_in x c_out

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_scale == "kaiming_uniform":
            kaiming_uniform_receptive_(self.coefficients, self.in_channels * self.kernel_size**2, a=5 ** 0.5, mode='fan_in', nonlinearity='relu')
        elif self.init_scale == "kaiming_normal":
            kaiming_normal_receptive_(self.coefficients, self.in_channels * self.kernel_size**2, a=5 ** 0.5, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # weight shape: c_in x c_out x self.kernel_size x self.kernel_size+
        w = F.conv2d(self.basis.view(-1, self.kernel_size, self.kernel_size).unsqueeze(0), self.coefficients, bias=None, padding=0).view(self.out_channels,
                                                                                              self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        if self.static_padding is not None:
            x = self.static_padding(x)
        return F.conv2d(x, w, bias=None, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        
class Signal_Decomposition(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, dilation=1, groups=1, static_padding=None, conv_kernel_size=3):
        super(Signal_Decomposition, self).__init__()
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = conv_kernel_size
        self.static_padding = static_padding
        self.conv_kernel_size = conv_kernel_size

        if kernel_size < conv_kernel_size:
            self.basis = torch.Tensor(normalize_basis(dct_2_basis(N=conv_kernel_size),conv_kernel_size))
            self.basis[:,kernel_size:conv_kernel_size] = 0
            self.basis[kernel_size:conv_kernel_size,:] = 0

        else:
            self.basis = torch.Tensor(normalize_basis(dct_2_basis(N=kernel_size),kernel_size))

        if conv_kernel_size < kernel_size:
            self.basis = self.basis[:conv_kernel_size, :conv_kernel_size, :conv_kernel_size, :conv_kernel_size].contiguous()
        
        self.basis_coef = torch.nn.Conv2d(in_channels, in_channels*self.kernel_size**2, self.kernel_size, bias=None,  padding=self.padding, stride=self.stride, dilation=self.dilation, groups=in_channels)
        self.basis_coef.weight = torch.nn.Parameter(self.basis.view(-1, 1, self.kernel_size, self.kernel_size).repeat(self.in_channels, 1, 1, 1), requires_grad=False)

        if self.groups != self.in_channels:    
            self.coefficients = torch.nn.Conv2d(in_channels*self.kernel_size**2, out_channels, 1, padding=0, bias=None)
        else:
            self.coefficients = torch.nn.Conv2d( in_channels*self.kernel_size**2, out_channels, 1, padding=0, bias=None, groups=groups) 
                

    def forward(self, x):
        if self.static_padding is not None:
            x = self.static_padding(x)
        x = self.basis_coef(x)
        x = self.coefficients(x)                                                                                      
        return x

