from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        episodic=False,
        mt_alpha=0.999,
        rst_m=0.01,
        ap=0.9,
        aug_num=32,
        noise_std=0.01,
        max_time_shift=8,
        scale_min=0.9,
        scale_max=1.1,
        channel_dropout_p=0.1,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.aug_num = int(aug_num)
        self.noise_std = float(noise_std)
        self.max_time_shift = int(max_time_shift)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.channel_dropout_p = float(channel_dropout_p)

    def _augment_eeg(self, x: torch.Tensor) -> torch.Tensor:
        """Lightweight EEG augmentations for CoTTA teacher averaging.

        The original CoTTA uses image augmentations. For EEG, we use simple
        amplitude/time/channel perturbations implemented in torch to avoid
        extra dependencies (PIL/torchvision).

        Input/Output
        ----------
        x: Tensor of shape (B, 1, C, T)
        """

        y = x
        # random amplitude scaling
        if self.scale_max > self.scale_min:
            scale = torch.empty((y.shape[0], 1, 1, 1), device=y.device).uniform_(self.scale_min, self.scale_max)
            y = y * scale
        # small time shift
        if self.max_time_shift > 0:
            shift = int(torch.randint(-self.max_time_shift, self.max_time_shift + 1, (1,), device="cpu").item())
            if shift != 0:
                y = torch.roll(y, shifts=shift, dims=-1)
        # gaussian noise
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        # channel dropout (same mask over time)
        if self.channel_dropout_p > 0:
            keep = (torch.rand((y.shape[0], 1, y.shape[2], 1), device=y.device) > self.channel_dropout_p).to(y.dtype)
            y = y * keep
        return y

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        outputs = self.model(x)[1]

        # Teacher prediction: weight-averaged teacher + (optional) augmentation-averaged predictions
        # following CoTTA Algorithm 1 and Eq.(3)(4)(5).
        with torch.no_grad():
            anchor_prob = torch.nn.functional.softmax(self.model_anchor(x)[1], dim=1).max(1)[0]
            standard_ema_logits = self.model_ema(x)[1]

            # Match the public CoTTA reference implementation:
            # average *logits* over augmentations, then softmax inside `softmax_entropy`.
            if self.aug_num > 0 and float(anchor_prob.mean().item()) < float(self.ap):
                logits_emas = []
                for _ in range(int(self.aug_num)):
                    x_aug = self._augment_eeg(x)
                    logits_aug = self.model_ema(x_aug)[1].detach()
                    logits_emas.append(logits_aug)
                outputs_ema = torch.stack(logits_emas, dim=0).mean(dim=0)
            else:
                outputs_ema = standard_ema_logits

        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.rst).float()
                        try:
                            mask = mask.cuda()
                        except:
                            mask = mask.cpu()
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:  # isinstance(m, nn.BatchNorm2d): collect all
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
