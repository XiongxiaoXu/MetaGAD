from turtle import distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def _construct_model_from_theta(theta, base_model, args):
    base_model_new = base_model.new().to(args.device)
    base_model_dict = base_model.state_dict()

    params, offset = {}, 0
    for k, v in base_model.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    base_model_dict.update(params)
    base_model_new.load_state_dict(base_model_dict)

    return base_model_new

def _hessian_vector_product(base_model, meta_model, vector, input_train, label_train, args, r=1e-2):
    bxent = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([args.pos_weight])).to(args.device)
    
    R = r / _concat(vector).norm()
    for p, v in zip(base_model.parameters(), vector):
        p.data.add_(R, v)
    logits = base_model(meta_model(input_train))
    loss = bxent(logits, label_train)
    grads_p = torch.autograd.grad(loss, meta_model.parameters())

    for p, v in zip(base_model.parameters(), vector):
        p.data.sub_(2*R, v)
    logits = base_model(meta_model(input_train))
    loss = bxent(logits, label_train)
    grads_n = torch.autograd.grad(loss, meta_model.parameters())

    for p, v in zip(base_model.parameters(), vector):
        p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

def step_metalearning_mlp(base_model, base_opt, 
                        meta_model, meta_opt,
                        eta, embedding,
                        idx_train, label_train, 
                        idx_val, label_val,
                        args):    
    bxent = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([args.pos_weight])).to(args.device)

    input_train = embedding[idx_train]
    input_val = embedding[idx_val]

    #update meta parameters
    meta_opt.zero_grad()

    #create unrolled model
    logits = base_model(meta_model(input_train))
    loss = bxent(logits, label_train)

    theta = _concat(base_model.parameters()).data

    try:
        moment = _concat(base_opt.state[v]['momentum_buffer'] for v in base_model.parameters()).mul_(args.momentum)
    except:
        moment = torch.zeros_like(theta)

    dtheta = _concat(torch.autograd.grad(loss, base_model.parameters())).data + args.weight_decay*theta
    unrolled_base_model = _construct_model_from_theta(theta.sub(eta, moment + dtheta), base_model, args)

    #calculate gradient
    unrolled_logits = unrolled_base_model(meta_model(input_val))
    unrolled_loss = bxent(unrolled_logits, label_val)
    unrolled_loss.backward()
    dalpha = [v.grad for v in meta_model.parameters()]
    vector = [v.grad.data for v in unrolled_base_model.parameters()]
    implicit_grads = _hessian_vector_product(base_model, meta_model, vector, input_train, label_train, args)

    for g, ig in zip(dalpha, implicit_grads):
        g.data.sub_(eta, ig.data)

    for v, g in zip(meta_model.parameters(), dalpha):
        v.grad.data.copy_(g.data)

    meta_opt.step()

    #update base parameters
    base_opt.zero_grad()
    logits = base_model(meta_model(input_train))
    loss = bxent(logits, label_train)
    loss.backward()
    # nn.utils.clip_grad_norm(base_model.parameters(), args.grad_clip)
    base_opt.step()

    return loss, unrolled_loss

