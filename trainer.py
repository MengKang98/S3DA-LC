import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import f1_score

from utility import Utility


class Trainer(object):
    def __init__(self, args, utility: Utility):
        self.utility = utility
        self.logger = utility.logger

        # common parameters
        self.current_iter = args.load_iter
        self.trgt = utility.trgt
        self.src = utility.src

        # get model
        self.network = utility.network
        self.optimizer = utility.optimizer

        # init src dataloaders
        self.src_loaders = {}
        for dom in self.src:
            self.src_loaders[dom] = utility.init_dataloader(dom)

        # parameters
        self.exp = args.exp
        self.tau = args.tau
        self.UTF = args.UTF
        self.w_k = args.w_k

        # init global variable
        self.class_threshold = None
        self.weight = torch.ones(1, len(self.src)).squeeze(0)
        self.sample_num = json.load(
            open(os.path.join(args.exp, args.dataset, "".join([args.dataset, ".json"])))
        )

    """
    Initialize pseudo target train data
    """

    def init_pseudo_trgt_dataloader(self):
        self.network.eval()
        self.logger.info(
            "-- Generate pseudo labels at {:6} --".format(self.current_iter)
        )
        with torch.no_grad():
            val_dataloaders = self.utility.init_dataloader(self.trgt, "train", False)

            all_index = []
            all_labels = []
            all_logits = [[] for _ in range(len(self.src))]

            for data in tqdm(val_dataloaders):
                index, images, labels = data
                images = images.to(self.utility.device).float()
                feats = self.network.model["F"](self.network.model["B"](images))
                logits = self.network.model["C"](feats, "single")

                for i in range(len(self.src)):
                    all_logits[i].append(logits[i])
                all_index.append(index)
                all_labels.append(labels)

            all_logits = torch.stack(
                [torch.cat(all_logits[i], dim=0).cpu() for i in range(len(all_logits))]
            )
            all_index = torch.cat(all_index, dim=0).cpu()
            all_labels = torch.cat(all_labels, dim=0).cpu()

            # ---------------------------- self.weight ---------------------------- #
            if self.w_k == 1:
                domain_wise_preds = all_logits.softmax(dim=2)
                domain_wise_ent = (
                    (-domain_wise_preds * torch.log(domain_wise_preds))
                    .sum(dim=2)
                    .mean(dim=1)
                )
                weight = 1 / domain_wise_ent
                weight = weight / weight.max()
                if self.current_iter == self.utility.pretrain_iter + 1:
                    self.weight = weight
                else:
                    self.weight = (
                        self.utility.pseudo_label[-1][1] * self.weight
                        + (1 - self.utility.pseudo_label[-1][1]) * weight
                    )

            # ----------------------- self.class_threshold ----------------------- #
            # prior distribution
            prior_dist = torch.stack(
                [torch.Tensor(self.sample_num[dom]) for dom in self.src]
            ).sum(dim=0)
            prior_dist = prior_dist / torch.sum(prior_dist)

            # distribution ensemble prediction
            ensemble_preds = torch.softmax(
                torch.mean(all_logits * self.weight.view(-1, 1, 1), dim=0), dim=1
            )
            pred_labels = torch.argmax(ensemble_preds, dim=1)
            predict_dist = torch.Tensor(
                [(pred_labels == i).sum() for i in range(self.utility.label_num)]
            )
            predict_dist = predict_dist / torch.sum(predict_dist)

            # learning difficulty
            learn_diff = predict_dist / prior_dist

            # normalize parameter: Phi
            Phi = max(learn_diff)
            if self.UTF > 0:
                Q1 = np.percentile(learn_diff, 25)
                Q3 = np.percentile(learn_diff, 75)
                Phi = Q3 + self.UTF * (Q3 - Q1)
                Phi = min(Phi, max(learn_diff))

            # class_threshold
            class_threshold = learn_diff / Phi
            class_threshold[class_threshold > 1.0] = 1.0
            class_threshold = self.tau * class_threshold
            if len(self.utility.pseudo_label) == 0:
                self.class_threshold = class_threshold
            else:
                self.class_threshold = (
                    self.utility.pseudo_label[-1][1] * self.class_threshold
                    + (1 - self.utility.pseudo_label[-1][1]) * class_threshold
                )

            # ------------------------- pseudo labels ---------------------------- #
            above_threshold = ensemble_preds > self.class_threshold
            for i in range(len(all_labels)):
                if not above_threshold[i][pred_labels[i]]:
                    pred_labels[i] = -1
            index = pred_labels > -1

            pseudo_labels = pred_labels[index]
            index = all_index[index]
            true_labels = all_labels[index]

        p_acc = f1_score(true_labels, pseudo_labels, average="micro")
        p_rate = float(len(true_labels) / len(all_labels))
        n_out = int(torch.sum(learn_diff > Phi))

        self.logger.info("{0:20} : {1:10}".format("Num of outliers", n_out))
        self.logger.info("{0:20} : {1:10f}".format("Pseudo acc", p_acc))
        self.logger.info("{0:20} : {1:10f}".format("Pseudo rate", p_rate))
        class_wise_num = []
        for i in range(self.utility.label_num):
            class_wise_num.append(int((pseudo_labels == i).sum()))

        self.utility.class_wise_num.append(class_wise_num)
        self.utility.pseudo_label.append([p_acc, p_rate, n_out, self.current_iter])
        self.utility.save_metrics()
        self.network.train()

        return self.utility.init_dataloader(
            self.trgt,
            mode=True,
            index=index,
            labels=pseudo_labels,
        )

    """
    Functions to calculate the loss value
    """

    def src_loss(self):
        return torch.nn.CrossEntropyLoss(reduction="mean")(
            torch.cat(self.src_logits, dim=0),
            torch.cat(self.src_labels, dim=0),
        )

    def trgt_loss(self):
        return torch.nn.CrossEntropyLoss(reduction="mean")(
            torch.cat(self.trgt_logits, dim=0),
            self.trgt_labels.repeat(len(self.src)),
        )

    """
    Functions for loading the data
    """

    def load_trgt_batch(self):
        try:
            _, images, labels = next(self.trgt_loader)
        except StopIteration:
            self.trgt_loader = self.init_pseudo_trgt_dataloader()
            _, images, labels = next(self.trgt_loader)

        self.inputs.append(Variable(images).to(self.utility.device).float())
        self.trgt_labels = Variable(labels).to(self.utility.device).long()

    def load_src_batches(self):
        for dom in self.src:
            try:
                _, images, labels = next(self.src_loaders[dom])
            except StopIteration:
                self.src_loaders[dom] = self.utility.init_dataloader(dom)
                _, images, labels = next(self.src_loaders[dom])

            self.inputs.append(Variable(images).to(self.utility.device).float())
            self.src_labels.append(Variable(labels).to(self.utility.device).long())

    """
    Functions for training on data
    """

    def pretrain(self):
        while self.current_iter <= self.utility.pretrain_iter:
            if self.current_iter % self.utility.val_after == 0:
                self.utility.evaluation(self.current_iter, self.weight)

            self.inputs = []
            self.src_labels = []
            self.load_src_batches()

            inputs = torch.cat(self.inputs, dim=0)
            feats = self.network.model["F"](self.network.model["B"](inputs)).chunk(
                len(self.src)
            )
            self.src_logits = self.network.model["C"](feats, "multi")

            self.optimizer.zero_grad()
            self.src_loss().backward()
            self.optimizer.step()
            self.current_iter += 1

    def adapt(self):
        self.trgt_loader = self.init_pseudo_trgt_dataloader()
        while self.current_iter <= self.utility.max_iter:
            if self.current_iter % self.utility.val_after == 0:
                self.utility.evaluation(self.current_iter, self.weight)

            self.inputs = []
            self.trgt_labels = []
            self.load_trgt_batch()
            self.src_labels = []
            self.load_src_batches()

            inputs = torch.cat(self.inputs, dim=0)
            feats = self.network.model["F"](self.network.model["B"](inputs)).chunk(
                len(self.src) + 1
            )
            self.src_logits = self.network.model["C"](feats[1:], "multi")
            self.trgt_logits = self.network.model["C"](feats[0], "single")

            loss = self.src_loss() + self.trgt_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.current_iter += 1

    def train(self):
        self.logger.info(f"============== Arguments ==============")
        self.logger.info(f">>> Dataset       : {self.utility.dataset}")
        self.logger.info(f">>> Task          : {self.utility.task}")
        self.logger.info(f">>> Backbone      : {self.utility.backbone}")
        self.logger.info(f">>> Pretrain iter : {self.utility.pretrain_iter}")
        self.logger.info(f">>> Val after     : {self.utility.val_after}")
        self.logger.info(f">>> Max iter      : {self.utility.max_iter}")
        self.logger.info(f">>> Device        : {self.utility.device}")
        self.logger.info(f"=======================================\n")

        self.pretrain()
        self.adapt()
