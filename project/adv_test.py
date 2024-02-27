import torch
import torch.nn as nn
import torchattacks
import os
import numpy as np
import random
import torchvision
from module.models.resnet import ResNet18
from module.models.lenet import LeNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def inverse_normalize(inputs, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(inputs.device)
    std = torch.tensor(std).reshape(1, 3, 1, 1).to(inputs.device)
    return inputs * std + mean

def convert_relu_to_softplus(model, beta):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta=beta))
        else:
            convert_relu_to_softplus(child, beta)

def load_model(model, activation_fn, softplus_beta):
    if model == "lenet":
        net = LeNet()
    elif model == "resnet18":
        net = ResNet18()
    elif model == "resnet18_imagenet100":
        net = torchvision.models.resnet18(num_classes=100)
    else:
        raise NameError(f"{model} is a wrong model")

    if activation_fn == "softplus":
        convert_relu_to_softplus(net, softplus_beta)

    return net

def safe_model_loader(model, ckpt):
    try:
        model.load_state_dict(ckpt["state_dict"])
    except:
        try:
            ckpt["state_dict"]["model.fc2.weight"] = ckpt["state_dict"]["model.fc.weight"]
            ckpt["state_dict"]["model.fc2.bias"] = ckpt["state_dict"]["model.fc.bias"]
            del ckpt["state_dict"]["model.fc.weight"], ckpt["state_dict"]["model.fc.bias"]
            model.load_state_dict(ckpt["state_dict"])

        except:
            for key1, key2 in zip(model.state_dict().keys() ,ckpt["state_dict"].keys()):
                try:
                    model.state_dict()[key1][:] = ckpt["state_dict"][key2]
                except:
                    model.state_dict()[key1] = ckpt["state_dict"][key2]

    return

def cli_main():
    # ------------
    # args
    # ------------
    exp_id = "normal_relu"
    device = torch.device("cuda")
    model = load_model("resnet18", "relu", 3.0).to(device)
    path = os.path.join("./output/cifar10_result", exp_id, "last.ckpt")
    ckpt = torch.load(path, map_location="cpu")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean, std)
    transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                    
                ]
            )
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    cifar_train = CIFAR10(root="/data/cifar10", train=True, transform=transform_train)
    cifar_test = CIFAR10(root="/data/cifar10", train=False, transform=transform_test)

    train_dataloader = DataLoader(
            cifar_train,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )
    
    test_dataloader = DataLoader(
            cifar_test,
            batch_size=100,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )
    

    safe_model_loader(model, ckpt)

    # attack
    model.eval()
    attack = torchattacks.PGD(model, random_start=False)
    attack.set_normalization_used(mean, std)

    num, num_ = 0,0
    z = None
    z_ = None
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        z = attack(x,y)

        c_out = model(x)
        output = model(z)

        # c_out = model(x)
        # output = model(z)

        _, y_ = torch.max(output,1)
        _, _y = torch.max(c_out,1)

        num_correct = (y==_y).sum().item()
        numcorrect = (y==y_).sum().item()

        num += numcorrect
        num_ += num_correct

        z_ = x

    print("模型准确率：", num_/len(test_dataloader.dataset))
    print("攻击成功率：", 1.0 - num/len(test_dataloader.dataset))

    img = inverse_normalize(z, mean, std)
    img = transforms.ToPILImage()(img[0].squeeze().cpu().detach())
    img.save("./adv.png")

    img = inverse_normalize(z_, mean, std)
    img = transforms.ToPILImage()(img[0].squeeze().cpu().detach())
    img.save("./nor.png")

if __name__ == "__main__":

    cli_main()
