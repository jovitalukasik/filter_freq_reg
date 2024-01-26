
import math
import torch
from dct_conv import Weight_Decomposition, Signal_Decomposition


def antidiagonals(L):
    h, w = len(L), len(L[0])
    return [torch.stack([L[p - q][q]
             for q in range(max(p-h+1,0), min(p+1, w))])
            for p in range(h + w - 1)]

def l2_reg(model, myhparams):
    l2_loss = 0
    l2_diff_loss = 0
    model_len = len([(m,n) for (m, n) in  filter(lambda t: type(t[1]) == Weight_Decomposition or type(t[1]) == Signal_Decomposition, model.named_modules())])
    l = 0
    for (module_name, dct_layer) in  filter(lambda t: type(t[1]) == Weight_Decomposition or type(t[1]) == Signal_Decomposition, model.named_modules()):
        add_penalty = 0
        add_diff_penalty = 0
        if type(dct_layer) == Weight_Decomposition:
            if dct_layer.kernel_size == 3:
                if myhparams.reg_all_freq:
                    for i in range(dct_layer.kernel_size):
                        for j in range(dct_layer.kernel_size):
                            if i == j == 0:
                                continue
                            add_penalty += torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,i,j], 2)*(i**2 + j**2)
                elif myhparams.reg_diff_all:
                    d_all = torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,:,:],2, dim=(0,1)).flatten().reshape(3,3)    
                    d_diag = antidiagonals(d_all)
                    diffs_all = torch.cat([d_diag[i-1][j]-d_diag[i] for i in range(1,len(d_diag)) for j in range(len(d_diag[i-1]))])
                    for diff in diffs_all:
                        if diff < 0:
                            add_diff_penalty += (-diff)
                else:
                    add_penalty += torch.sum(torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,:2,2],2, dim=(0,1))*2)
                    add_penalty += torch.sum(torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,2,:3],2, dim=(0,1))*2)

                if myhparams.reg_diff:
                    d_0 =  torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,:2,:2],2, dim=(0,1)).flatten()[0]
                    d_1 = torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,:2,:2],2, dim=(0,1)).flatten()[1:]
                    diffs_all = d_0-d_1
                    for diff in diffs_all:
                        if diff < 0:
                            add_diff_penalty += (-diff)
            else:
                if myhparams.reg_all_freq:
                    for i in range(5):
                        for j in range(5):
                            if (i+j) in [0,1] or i==j==1:
                                continue
                            add_penalty += torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,i,j], 2)*(i**2 + j**2)
                elif myhparams.reg_diff_all:
                    d_all = torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,:,:],2, dim=(0,1)).flatten().reshape(5,5)    
                    d_diag = antidiagonals(d_all)
                    diffs_all = torch.cat([d_diag[i-1][j]-d_diag[i] for i in range(1,len(d_diag)) for j in range(len(d_diag[i-1]))])
                    for diff in diffs_all:
                        if diff < 0:
                            add_diff_penalty += (-diff)
                else:
                    for i in range(3,5):
                        add_penalty += torch.sum(torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,:i,i],2, dim=(0,1))*i)
                        add_penalty += torch.sum(torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,i,:i+1],2, dim=(0,1))*i)
            
                if myhparams.reg_diff:
                    d_0 =  torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,:2,:2],2, dim=(0,1)).flatten()[0]
                    d_1 = torch.norm(dct_layer.coefficients.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,:2,:2],2, dim=(0,1)).flatten()[1:]
                    diffs_all = d_0-d_1
                    for diff in diffs_all:
                        if diff < 0:
                            add_diff_penalty += (-diff)

        else:
            if dct_layer.kernel_size == 3:
                if myhparams.reg_all_freq:
                    for i in range(dct_layer.kernel_size):
                        for j in range(dct_layer.kernel_size):
                            if i == j == 0:
                                continue
                            add_penalty += torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,i,j], 2)*(i**2 + j**2)
                elif myhparams.reg_diff_all:
                    d_all = torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,:,:],2, dim=(0,1)).flatten().reshape(3,3)    
                    d_diag = antidiagonals(d_all)
                    diffs_all = torch.cat([d_diag[i-1][j]-d_diag[i] for i in range(1,len(d_diag)) for j in range(len(d_diag[i-1]))])
                    for diff in diffs_all:
                        if diff < 0:
                            add_diff_penalty += (-diff)
                else:
                    add_penalty += torch.sum(torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,:2,2], 2, dim=(0,1))*2)
                    add_penalty += torch.sum(torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,2,:3], 2, dim=(0,1))*2)
                
                if myhparams.reg_diff:
                    d_0 =  torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,:2,:2],2, dim=(0,1)).flatten()[0]
                    d_1 = torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 3, 3)[:,:,:2,:2],2, dim=(0,1)).flatten()[1:]
                    diffs_all = d_0-d_1
                    for diff in diffs_all:
                        if diff < 0:
                            add_diff_penalty += (-diff)
            
            else:
                if myhparams.reg_all_freq:
                    for i in range(5):
                        for j in range(5):
                            if (i+j) in [0,1] or i==j==1:
                                continue
                            add_penalty += torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,i,j], 2)*(i**2 + j**2)
                elif myhparams.reg_diff_all:
                    d_all = torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,:,:],2, dim=(0,1)).flatten().reshape(5,5)    
                    d_diag = antidiagonals(d_all)
                    diffs_all = torch.cat([d_diag[i-1][j]-d_diag[i] for i in range(1,len(d_diag)) for j in range(len(d_diag[i-1]))])
                    for diff in diffs_all:
                        if diff < 0:
                            add_diff_penalty += (-diff)
                else:
                    for i in range(3,5):
                        add_penalty += torch.sum(torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,:i,i],2, dim=(0,1))*i)
                        add_penalty += torch.sum(torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,i,:i+1],2, dim=(0,1))*i)

                if myhparams.reg_diff:
                    d_0 =  torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,:2,:2],2, dim=(0,1)).flatten()[0]
                    d_1 = torch.norm(dct_layer.coefficients.weight.reshape(dct_layer.out_channels, dct_layer.in_channels//dct_layer.groups, 5, 5)[:,:,:2,:2],2, dim=(0,1)).flatten()[1:]
                    diffs_all = d_0-d_1
                    for diff in diffs_all:
                        if diff < 0:
                            add_diff_penalty += (-diff)
                
        l2_loss += add_penalty
        l2_diff_loss += add_diff_penalty
        l += 1
        if l >  model_len // myhparams.reg_depth:
            break
    return l2_loss * myhparams.l2_lambda +  l2_diff_loss* myhparams.l2_lambda

