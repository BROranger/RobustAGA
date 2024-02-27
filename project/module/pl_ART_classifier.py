import torch
import torch.nn.functional as F

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from module import LitClassifier


class LitAdaptiveRobustEXplanationTrainClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def default_step(self, x, y, stage):
        criterion_fn = lambda output, y: output[range(len(y)), y].sum()
        with torch.enable_grad():
            images = x.clone().detach().to(self.device)
            labels = y.clone().detach().to(self.device)

            x.requires_grad = True
            y_hat = self(x)
            yc_hat = criterion_fn(y_hat, y)
            acc = self.metric.get_accuracy(y_hat, y)
            grad_s = torch.autograd.grad(outputs=yc_hat, inputs=x, create_graph=True, retain_graph=True)[0]
            ce_loss = F.cross_entropy(y_hat, y, reduction="mean")

            # DeepFool 算法找到离模型边界最近的样本
            batch_size = len(images)
            correct = torch.tensor([True] * batch_size)
            target_labels = labels.clone().detach().to(self.device)
            curr_steps = 0

            adv_images = []
            for idx in range(batch_size):
                image = images[idx : idx + 1].clone().detach()
                adv_images.append(image)

            while (True in correct) and (curr_steps < self.hparams.step_size):
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

            adv_images.requires_grad = True
            yhat_adv = self(adv_images)
            yc_hat_adv = criterion_fn(yhat_adv, y)
            grad_adv = torch.autograd.grad(outputs=yc_hat_adv, inputs=adv_images, create_graph=True, retain_graph=True)[0]

            true_adv_grads, false_adv_grads = [], []
            true_s_grads, false_s_grads = [], []
            for idx in range(batch_size):
                if correct[idx] == True:
                    true_adv_grads.append(grad_adv[idx])
                    true_s_grads.append(grad_s[idx])
                else:
                    false_adv_grads.append(grad_adv[idx])
                    false_s_grads.append(grad_s[idx])
            if true_adv_grads and false_adv_grads:
                true_adv_grads = torch.cat(true_adv_grads).detach()
                false_adv_grads = torch.cat(false_adv_grads).detach()
                true_s_grads = torch.cat(true_s_grads).detach()
                false_s_grads = torch.cat(false_s_grads).detach()
            elif not true_adv_grads:
                true_adv_grads = torch.tensor([0]*batch_size,dtype=torch.float32).unsqueeze(dim=1)
                true_s_grads = torch.tensor([0]*batch_size,dtype=torch.float32).unsqueeze(dim=1)
                false_adv_grads = torch.cat(false_adv_grads).detach()
                false_s_grads = torch.cat(false_s_grads).detach()
            else:
                true_adv_grads = torch.cat(true_adv_grads).detach()
                true_s_grads = torch.cat(true_s_grads).detach()
                false_adv_grads = torch.tensor([0]*batch_size,dtype=torch.float32).unsqueeze(dim=1)
                false_s_grads = torch.tensor([0]*batch_size,dtype=torch.float32).unsqueeze(dim=1)


            t_cossim = self.metric.calc_cossim(true_adv_grads, true_s_grads).mean()
            t_cosd = (1 - t_cossim) / 2

            f_cossim = self.metric.calc_cossim(false_adv_grads, false_s_grads).mean()
            f_cosd = (1 - f_cossim) / 2

            t_l2 = (true_adv_grads - true_s_grads).flatten(start_dim=1).norm(dim=1).square().mean()
            f_l2 = (false_adv_grads - false_s_grads).flatten(start_dim=1).norm(dim=1).square().mean()

            t_reg_loss = self.hparams.lamb_l2 * t_l2 + self.hparams.lamb_cos * t_cosd
            f_reg_loss = self.hparams.lamb_l2 * f_l2 + self.hparams.lamb_cos * f_cosd
            reg_loss = t_reg_loss - f_reg_loss

            loss = ce_loss + reg_loss


        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_ce_loss": ce_loss, f"{stage}_robust_loss": reg_loss, f"{stage}_acc": acc},
            prog_bar=True,
            sync_dist=True,
        )

        return loss
    
    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self(image)[0]
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
        value = torch.abs(f_prime) / torch.norm(torch.nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (
            torch.abs(f_prime[hat_L])
            * w_prime[hat_L]
            / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (1 + self.hparams.overshoot) * delta
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
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifier.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("ART classifier")
        group.add_argument("--step_size", type=float, default=30)
        group.add_argument("--overshoot", type=int, default=0.02)
        group.add_argument("--lamb_l2", type=float, default=1.0)
        group.add_argument("--lamb_cos", type=float, default=1.0)
        
        return parser