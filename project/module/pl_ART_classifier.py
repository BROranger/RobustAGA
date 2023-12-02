import torch
import torch.nn.functional as F

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from module import LitClassifier


class LitAdaptiveRobustEXplanationTrainClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def default_step(self, x, y, stage):

        criterion_kl = torch.nn.KLDivLoss(size_average=False)
        batch_size = len(x)
        # generate adversarial example
        x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
        if self.hparams.distance == 'l_inf':
            for _ in range(self.hparams.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(self(x_adv), dim=1),
                                        F.softmax(self(x), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.hparams.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x - self.hparams.epsilon), x + self.hparams.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif self.hparams.distance == 'l_2':
            delta = 0.001 * torch.randn(x.shape).detach()
            delta = torch.autograd.Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = torch.optim.SGD([delta], lr=self.hparams.epsilon / self.hparams.perturb_steps * 2)

            for _ in range(self.hparams.perturb_steps):
                adv = x + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(self(adv), dim=1),
                                            F.softmax(self(x), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x)
                delta.data.clamp_(0, 1).sub_(x)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.hparams.epsilon)
            x_adv = torch.autograd.Variable(x + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = torch.autograd.Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # calculate robust loss
        logits = self(x)
        acc = self.metric.get_accuracy(logits, y)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(self(x_adv), dim=1),
                                                        F.softmax(self(x), dim=1))
        loss = loss_natural + self.hparams.beta * loss_robust


        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_ce_loss": loss_natural, f"{stage}_robust_loss": loss_robust, f"{stage}_acc": acc,},
            prog_bar=True,
            sync_dist=True,
        )

        return loss
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifier.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("AT classifier")
        group.add_argument("--epsilon", type=float, default=8/255)
        group.add_argument("--distance", type=str, default="l_inf")
        group.add_argument("--step_size", type=float, default=2/255)
        group.add_argument("--perturb_steps", type=int, default=20)
        group.add_argument("--beta", type=float, default=1.0)
        
        return parser

    def forward_return_target_labels(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx : idx + 1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]:
                    continue
                early_stop, pre, adv_image = self._forward_indiv(
                    adv_images[idx], labels[idx]
                )
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()
        return adv_images, target_labels

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.get_logits(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (
            torch.abs(f_prime[hat_L])
            * w_prime[hat_L]
            / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (1 + self.overshoot) * delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)