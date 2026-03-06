from torch.optim.lr_scheduler import _LRScheduler


class WarmupPolyLRScheduler(_LRScheduler):
    """Learning rate scheduler with linear warmup followed by polynomial decay."""

    def __init__(
        self,
        optimizer,
        initial_lr,
        max_lr,
        max_steps,
        warmup_steps,
        exponent=0.9,
        current_step=None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:
            # Linear warmup from initial_lr to max_lr
            alpha = current_step / self.warmup_steps
            new_lr = self.initial_lr + alpha * (self.max_lr - self.initial_lr)
        else:
            # Poly decay from max_lr
            # For poly decay we usually want (1 - iter/max_iter)^exp
            # Here we map [warmup_steps, max_steps] to [0, 1] relative to decay phase
            effective_step = current_step - self.warmup_steps
            effective_max_steps = self.max_steps - self.warmup_steps

            if effective_max_steps > 0:
                # clip effective_step to not exceed effective_max_steps handling training beyond max_epochs
                effective_step = min(effective_step, effective_max_steps)
                new_lr = (
                    self.max_lr
                    * (1 - effective_step / effective_max_steps) ** self.exponent
                )
            else:
                new_lr = self.max_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
