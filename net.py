import torch.nn as nn
from torchvision import models
from torch.autograd import Function


class Net(nn.Module):
    def __init__(self, utility):
        super(Net, self).__init__()
        self.model = {
            "B": BackBoneLayer(utility.backbone, utility.output_dims),
            "F": ForwardLayer(
                utility.output_dims,
                utility.output_dims // 2,
                utility.feats_dims,
            ),
            "C": Classifiers(utility.feats_dims, len(utility.src), utility.label_num),
        }
        for name, module in self.model.items():
            self.add_module(name, module)

    def forward(self, _):
        raise NotImplementedError("Implemented a custom forward in train loop")


# classifier
class Classifiers(nn.Module):
    def __init__(self, feat_dims, domain_num, class_num):
        super(Classifiers, self).__init__()
        self.domain_num = domain_num
        self.class_num = class_num
        self.feat_dims = feat_dims
        self.modlist = nn.ModuleList(
            [nn.Linear(self.feat_dims, self.class_num) for _ in range(domain_num)]
        )

    def forward(self, feats, mode):
        logits = []
        if mode == "multi":
            for i in range(len(self.modlist)):
                logits.append(self.modlist[i](feats[i]))
        else:
            for linear in self.modlist:
                logits.append(linear(feats))
        return logits


# forward layer
class ForwardLayer(nn.Module):
    def __init__(self, output_dims, mid_dims, feat_dims):
        super(ForwardLayer, self).__init__()
        self.output_dims = output_dims
        self.mid_dims = mid_dims
        self.feat_dims = feat_dims

        self.forwardlayer = nn.Sequential(
            nn.Linear(self.output_dims, self.mid_dims),
            nn.ELU(),
            nn.Linear(self.mid_dims, self.mid_dims),
            nn.BatchNorm1d(self.mid_dims),
            nn.ELU(),
            nn.Linear(self.mid_dims, self.feat_dims),
            nn.ELU(),
            nn.Linear(self.feat_dims, self.feat_dims),
            nn.BatchNorm1d(self.feat_dims),
            nn.ELU(),
        )

    def forward(self, x):
        return self.forwardlayer(x)


# backbone layer
class BackBoneLayer(nn.Module):
    def __init__(self, backbone, output_dims):
        super(BackBoneLayer, self).__init__()

        if backbone == "resnet101":
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*[x for x in list(resnet.children())[:-1]])
        self.output_dims = output_dims

    def forward(self, x):
        return self.backbone(x).view((x.shape[0], self.output_dims))


if __name__ == "__main__":
    raise NotImplementedError("Please check README.md for execution details")
