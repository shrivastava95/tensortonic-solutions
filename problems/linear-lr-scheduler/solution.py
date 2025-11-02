def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    if warmup_steps != 0 and step < warmup_steps:
        lr = step * initial_lr / warmup_steps
    elif step < total_steps:
        if total_steps != warmup_steps:
            lr = final_lr + (initial_lr - final_lr) * (total_steps - step) / (total_steps - warmup_steps)
    else:
        lr = final_lr
    return lr
