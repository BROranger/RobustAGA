import torch
import torch.nn.functional as F

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from module import LitClassifier


class LitTradesTrainClassifier(LitClassifier):
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
        group.add_argument("--perturb_steps", type=int, default=10)
        group.add_argument("--beta", type=float, default=1.0)
        
        return parser
