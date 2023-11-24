from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from . import LitClassifier
from module.lrp_module.load_model import load_model
from module.utils.interpreter import Interpreter
import torch.nn.functional as F


class LitClassifierAdvTester(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = self.model(x)
        return x

    def test_step(self, batch, batch_idx):
        x_s, y_s = batch
        yhat_s = self(x_s)

        # get adv imgs
        x_adv = self.get_adv_img(x_s, y_s).detach().requires_grad_()
        yhat_adv = self(x_adv)

        # metrics
        acc_adv = self.metric.get_accuracy(yhat_adv, y_s)
        acc = self.metric.get_accuracy(yhat_s, y_s)

        prefix = (
            f"adv_eps_{self.hparams.train_epsilon}_iter_{self.hparams.train_perturb_steps}"
        )

        # log results
        self.log_dict(
            {f"{prefix}_acc_adv": acc_adv, f"{prefix}_acc_nor":acc}, prog_bar=True, sync_dist=True,
        )


    def get_adv_img(self, x, y):
        with torch.enable_grad():
            x = x.requires_grad_()

            # generate adversarial example
            x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
            if self.hparams.train_distance == 'l_inf':
                for _ in range(self.hparams.train_perturb_steps):
                    x_adv.requires_grad_()
                   
                    y_hat = self(x_adv) 
                    cost = F.cross_entropy(y_hat, y)
                    grad = torch.autograd.grad(cost, [x_adv])[0]

                    x_adv = x_adv.detach() + self.hparams.train_step_size * torch.sign(grad.detach())
                    x_adv = torch.min(torch.max(x_adv, x - self.hparams.train_epsilon), x + self.hparams.train_epsilon)
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv.detach()


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifier.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Adversarial examples test")
        group.add_argument("--train_epsilon", type=float, default=8/255)
        group.add_argument("--train_distance", type=str, default="l_inf")
        group.add_argument("--train_step_size", type=float, default=2/255)
        group.add_argument("--train_perturb_steps", type=int, default=20)
        group.add_argument("--train_beta", type=float, default=1.0)
        return parser
