from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from . import LitClassifier
from module.lrp_module.load_model import load_model
from module.utils.interpreter import Interpreter
import torch.nn.functional as F
import torchattacks

class LitClassifierAdvTester(LitClassifier):
    def __init__(self, **kwargs, ):
        super().__init__(**kwargs)
        self.interpreter = Interpreter(self.model)
        # self.attack = torchattacks.PGD(self.model.cuda(), random_start=False)
        # self.attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))



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

            # if yhat_s

            h_adv = self.interpreter.get_heatmap(
                x_adv, y_s, yhat_adv, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
            ).detach()

            # metrics
            acc_adv = self.metric.get_accuracy(yhat_adv, y_s)
            acc = self.metric.get_accuracy(yhat_s, y_s)

            # 根据mask，筛选出相应的 x_adv x_s  yhat_adv  y_s h_s h_adv
            _, y_adv = torch.max(yhat_adv, 1)
            _, y_ps = torch.max(yhat_s, 1)
            mask = (y_adv != y_ps) 
            true_h_s, true_h_adv = [], []
            true_adv, true_x_s = [], []
            for index, s in enumerate(mask):
                if s == True:
                    true_h_s.append(h_s[index])
                    true_h_adv.append(h_adv[index])
                    true_adv.append(x_adv[index])
                    true_x_s.append(x_s[index])

            true_adv = torch.stack(true_adv)
            true_h_adv = torch.stack(true_h_adv)
            true_h_s = torch.stack(true_h_s)
            true_x_s = torch.stack(true_x_s)


            prefix = (
                f"adv_eps_{self.hparams.test_epsilon}_iter_{self.hparams.test_perturb_steps}"
            )

            # log results
            self.log_dict(
                {f"{prefix}_acc_adv": acc_adv, f"{prefix}_acc_nor":acc}, prog_bar=True, sync_dist=True,
            )
            self.log_hm_metrics(h_adv, h_s, f"{prefix}_(h_a,h_s)")
            self.log_hm_metrics(true_h_adv, true_h_s, "successful_attack_adv")


    # def get_adv_img(self, x, y):
    #     with torch.enable_grad():
    #         x = x.requires_grad_()
    #         x_adv = self.attack(x,y)

    #     return x_adv.detach()

    def get_adv_img(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = torch.nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        # if self.random_start:
        #     # Starting at a uniformly random point
        #     adv_images = adv_images + torch.empty_like(adv_images).uniform_(
        #         -self.eps, self.eps
        #     )
        #     adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.hparams.test_perturb_steps):
            adv_images.requires_grad = True
            outputs = self(adv_images)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.hparams.test_step_size * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.hparams.test_epsilon, max=self.hparams.test_epsilon)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
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
        group.add_argument("--test_epsilon", type=float, default=8/255)
        group.add_argument("--test_distance", type=str, default="l_inf")
        group.add_argument("--test_step_size", type=float, default=2/255)
        group.add_argument("--test_perturb_steps", type=int, default=20)
        group.add_argument("--test_beta", type=float, default=1.0)

        group.add_argument("--hm_method", type=str, default="grad", help="interpretation method")
        group.add_argument("--hm_norm", type=str, default="standard")
        group.add_argument("--hm_thres", type=str, default="abs")
        return parser
