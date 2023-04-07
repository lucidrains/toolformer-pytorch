from torch.optim import AdamW, Adam

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

def get_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True,
    **kwargs
):
    has_weight_decay = wd > 0

    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if group_wd_params and has_weight_decay:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    adam_kwargs = dict(lr = lr, betas = betas, eps = eps)

    if not has_weight_decay:
        return Adam(params, **adam_kwargs)

    return AdamW(params, weight_decay = wd, **adam_kwargs)
