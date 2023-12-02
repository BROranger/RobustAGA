from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from . import LitClassifier
from module.lrp_module.load_model import load_model
from module.utils.interpreter import Interpreter
import torch.nn.functional as F


class LitClassifierAdvTester(LitClassifier):
    def __init__(self, **kwargs, ):
        super().__init__(**kwargs)
        self.interpreter = Interpreter(self.model)

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1).cuda()

    def forward(self, x):
        x = self.model(x)
        return x

    def test_step(self, batch, batch_idx):
        x_s, y_s = batch
        with torch.enable_grad():
            x_s = x_s.requires_grad_()
            yhat_s = self(x_s)
            h_s = self.interpreter.get_heatmap(
                x_s, y_s, yhat_s, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
            ).detach()

            # get adv imgs
            x_adv = self.get_adv_img(x_s, y_s).detach().requires_grad_()
            yhat_adv = self(x_adv)
            h_adv = self.interpreter.get_heatmap(
                x_adv, y_s, yhat_adv, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
            ).detach()

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
            self.log_hm_metrics(h_adv, h_s, f"{prefix}_(h_a,h_s)")


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
    
    def log_hm_metrics(self, h1, h2, name):
        loss = F.mse_loss(h1, h2, reduction="sum") / h1.shape[0]
        pcc = self.metric.calc_pcc(h1, h2)
        ssim = self.metric.calc_ssim(h1, h2)
        cossim = self.metric.calc_cossim(h1, h2)

        h1 = h1 / (h1.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        h2 = h2 / (h2.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        sumnorm_ssim = self.metric.calc_ssim(h1, h2)
        # log results
        self.log_dict(
            {
                f"{name}_mse": loss,
                f"{name}_pcc": pcc,
                f"{name}_ssim": ssim,
                f"{name}_cossim": cossim,
                f"{name}_sumnorm_ssim": sumnorm_ssim,
            },
            prog_bar=True,
            sync_dist=True,
        )
        return


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

        group.add_argument("--hm_method", type=str, default="grad", help="interpretation method")
        group.add_argument("--hm_norm", type=str, default="standard")
        group.add_argument("--hm_thres", type=str, default="abs")
        return parser
