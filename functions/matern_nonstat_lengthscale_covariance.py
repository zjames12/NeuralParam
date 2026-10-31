from __future__ import annotations

import math

import torch

class MaternNonstatLengthscaleCovariance(torch.autograd.Function):                                                                      
    @staticmethod                                                                                                     
    def forward(ctx, x1, x2, lf, nu, dist_func):                                                             
        if any(ctx.needs_input_grad[:2]):                                                                             
            raise RuntimeError("MaternNonstatLengthscaleCovariance cannot compute gradients with respect to x1 and x2")                 
        # if lengthscale.size(-1) > 1:
        #     raise ValueError("MaternNonstatLengthscaleCovariance cannot handle multiple lengthscales")                                  
        l1 = lf(x1)
        l2 = lf(x2)
        mean = x1.mean(dim=-2, keepdim=True)
        x1_ = x1 - mean                                                                           
        x2_ = x2 - mean
        # scaled_unitless_dist = dist_func(x1_, x2_).mul_(math.sqrt(2 * nu))                                            
        scaled_unitless_dist = dist_func(x1_, x2_).mul_(2*math.sqrt(2 * nu)).div_(l1 + l2.T)
        if nu == 0.5:                                                                                                 
            covar_mat = scaled_unitless_dist.neg_().exp_()
        elif nu == 1.5:                                                                                               
            linear_term = scaled_unitless_dist.clone().add_(1)
            exp_component = scaled_unitless_dist.neg_().exp_()                                                        
            covar_mat = linear_term.mul_(exp_component)
        elif nu == 2.5:                                                                                               
            linear_term = scaled_unitless_dist.clone().add_(1)
            quadratic_term = scaled_unitless_dist.clone().pow_(2).div_(3)                                             
            exp_component = scaled_unitless_dist.neg_().exp_()
            covar_mat = exp_component.mul_(linear_term.add_(quadratic_term))

        return covar_mat.mul_(torch.sqrt(2 * l1 * l2.T))

    @staticmethod
    def backward(ctx, grad_output):
        # Let PyTorch handle gradients automatically
        return None, None, None, None, None



    
