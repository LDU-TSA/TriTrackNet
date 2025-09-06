import torch
from torch.optim import Optimizer


class perturbopt(Optimizer):
    """
    SAM with Adversarial Training: Sharpness-Aware Minimization combined with adversarial training.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, steps=3, epsilon=1.9, **kwargs):
        """
        :param steps: 扰动的步骤数，表示每次优化时扰动的增加步骤
        :param epsilon: 对抗扰动的强度
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, steps=steps, epsilon=epsilon, **kwargs)
        super(perturbopt, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def generate_adversarial_example(self, model, inputs, targets, epsilon):
        """
        生成对抗样本
        """
        inputs.requires_grad = True
        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        model.zero_grad()
        loss.backward()

        # 获取对抗扰动
        adversarial_perturbation = epsilon * inputs.grad.sign()
        return inputs + adversarial_perturbation

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        # 计算当前步数的扰动强度
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 计算扰动量，根据步骤进行逐渐增大的扰动
                e_w = (
                        (torch.pow(p, 2) if group["adaptive"] else 1.0)
                        * p.grad
                        * scale.to(p)
                )

                # 增加扰动强度：step越大，扰动越强
                e_w *= (self.param_groups[0]["epsilon"] * (self.param_groups[0]["steps"] - 1) / self.param_groups[0][
                    "steps"])

                p.add_(e_w)  # 对模型参数进行扰动
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, model, inputs, targets, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # 让闭包函数执行前向反向传播

        # 生成对抗样本
        adversarial_inputs = self.generate_adversarial_example(model, inputs, targets, self.param_groups[0]['epsilon'])

        # 对抗样本用于前向和反向传播
        outputs = model(adversarial_inputs)  # 使用对抗样本进行前向传播
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)  # 计算损失

        # 反向传播
        model.zero_grad()
        loss.backward()

        # 使用steps参数控制逐步扰动
        for step in range(self.param_groups[0]["steps"]):
            self.first_step(zero_grad=True)  # 在每步更新中进行扰动
            closure()  # 执行前向反向传播
            self.second_step()

        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=3)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm
