import os
import json
import torch
import logging
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from net import Net
from config import settings
from dataset import ImageList
from dataset import create_index


class Utility:
    def __init__(self, args: argparse):
        #  common parameters
        self.dataset = args.dataset
        self.task = args.task
        self.backbone = "resnet101" if args.dataset == "domain-net" else "resnet50"
        self.path = args.path
        self.exp = args.exp

        self.filename = (
            args.exp
            + "("
            + args.no
            + ")_UTF("
            + str(args.UTF)
            + ")_tau("
            + str(args.tau)
            + ")_w_k("
            + str(args.w_k)
            + ")_seed("
            + str(args.seed)
            + ")"
        )

        # train parameters
        self.worker = args.worker
        self.val_after = settings[args.dataset]["val_after"]
        self.batch_size = settings[args.dataset]["batch_size"]
        self.val_batch_size = settings[args.dataset]["val_batch_size"]
        self.pretrain_iter = settings[args.dataset][args.task]["pretrain_iter"]
        self.max_iter = settings[args.dataset]["max_iter"]

        # dataset parameters
        self.label_num = settings[args.dataset]["label_num"]
        self.src = settings[args.dataset][args.task]["src"]
        self.trgt = settings[args.dataset][args.task]["trgt"]
        self.domains = [self.trgt]
        self.domains.extend(self.src)

        # init file directions
        self.index_dir = os.path.join(args.exp, args.dataset, "index")
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            create_index(self)
        self.exp_dir = os.path.join(
            args.exp, args.dataset, "_".join([args.task, self.backbone])
        )
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        # init logger
        self.no = args.no
        self.log_dir = os.path.join("logs", args.dataset, args.task)
        self.logger = self.init_logger()

        # network parameters
        self.output_dims = 2048
        self.feats_dims = 256

        # optimizer parameter
        self.lr = 1e-5
        self.weight_decay = 5e-4

        # init model
        self.device = "cuda:" + str(args.gpu)
        self.network = Net(self).to(self.device)
        self.optimizer = self.init_optimizers()
        if args.load_iter > 0:
            self.logger.info("Load weights from iter: {}\n".format(args.load_iter))
            self.load_weights(args.load_iter)

        # weight saving parameter
        self.last_record_iter = -1

        # josn record
        self.pretrain_acc = 0
        self.ensemble_acc = []
        self.pseudo_label = []
        self.class_wise_num = []
        self.class_wise_predict = []
        self.domain_acc = {d: [] for d in self.src}

    """
        Init Logger
    """

    def init_logger(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        formatter = logging.Formatter("[%(asctime)-15s]%(message)s")

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            os.path.join(self.log_dir, self.filename + ".txt"), mode="w"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        return logger

    """
        Saving weights
    """

    def check_and_save_weights(self, current_iter):
        if max(self.ensemble_acc) == self.ensemble_acc[-1]:
            if self.last_record_iter >= 0:
                os.remove(
                    os.path.join(
                        self.exp_dir,
                        "model_" + str(self.last_record_iter) + ".pth",
                    )
                )
                os.remove(
                    os.path.join(
                        self.exp_dir,
                        "opt_" + str(self.last_record_iter) + ".pth",
                    )
                )
            self.last_record_iter = current_iter

            self.logger.info("Saving best weight at iter {}".format(current_iter))
            self.save_weights(str(current_iter))
        if current_iter == self.pretrain_iter:
            self.save_weights("pretrain_" + str(current_iter))

    def save_weights(self, iter):
        model_state_dict = {}
        for name, module in self.network.model.items():
            model_state_dict[name] = module.state_dict()
        torch.save(
            model_state_dict,
            os.path.join(self.exp_dir, "model_" + iter + ".pth"),
        )

        optimizer_state = self.optimizer.state_dict()
        torch.save(
            optimizer_state,
            os.path.join(self.exp_dir, "opt_" + iter + ".pth"),
        )

    """
        Load weights
    """

    def load_weights(self, load_iter):
        file_suffix = str(load_iter) + ".pth"
        if load_iter == self.pretrain_iter:
            file_suffix = "pretrain_" + file_suffix
        model_path = os.path.join(self.exp_dir, "model_" + file_suffix)
        model_state_dict = torch.load(model_path, map_location=self.device)
        for name, module in self.network.model.items():
            module.load_state_dict(model_state_dict[name])

        opt_path = os.path.join(self.exp_dir, "opt_" + file_suffix)
        optimizer_state = torch.load(opt_path, map_location=self.device)
        self.optimizer.load_state_dict(optimizer_state)

    """
        Initializing optimizers
    """

    def init_optimizers(self):
        opt_param_list = []
        for name, module in self.network.model.items():
            if name == "B":
                opt_param_list.append(
                    {
                        "params": module.parameters(),
                        "lr": self.lr / 10.0,
                        "weight_decay": self.weight_decay,
                    }
                )
            else:
                opt_param_list.append(
                    {
                        "params": module.parameters(),
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
        return optim.Adam(params=opt_param_list)

    """
        Initialize dataloaders
    """

    def init_dataloader(self, dom, file="train", mode=True, index=None, labels=None):
        dataset = ImageList(
            index_file_dir=os.path.join(
                self.index_dir,
                "_".join([dom, file + ".npy"]),
            ),
            aug=mode,
            index=index,
            labels=labels,
        )
        return iter(
            DataLoader(
                dataset,
                batch_size=self.batch_size if mode else self.val_batch_size,
                shuffle=mode,
                num_workers=self.worker,
                drop_last=mode,
                pin_memory=True,
            )
        )

    """
        Val on all domain
    """

    def val(self, current_iter, weight):
        val_dataloaders = self.init_dataloader(self.trgt, "test", False)
        with torch.no_grad():
            all_labels = []
            all_logits = [[] for _ in range(len(self.src))]
            domain_wise_preds = []

            for data in tqdm(val_dataloaders):
                _, images, labels = data

                images = images.to(self.device).float()
                feats = self.network.model["F"](self.network.model["B"](images))
                logits = self.network.model["C"](feats, "single")

                for i in range(len(self.src)):
                    all_logits[i].append(logits[i])
                all_labels.append(labels)

            all_preds = 0
            for i in range(len(self.src)):
                all_logits[i] = torch.cat(all_logits[i], dim=0).cpu()
                # ensemble prediction
                all_preds += weight[i] * all_logits[i]
                # domain-wise prediction
                domain_wise_preds.append(torch.argmax(all_logits[i], dim=1))
            all_labels = torch.cat(all_labels, dim=0).cpu()
            all_preds = torch.argmax(all_preds, dim=1)

            ens_acc = f1_score(all_labels, all_preds, average="micro")
            if current_iter >= self.pretrain_iter:
                class_wise_predict = []
                for i in range(self.label_num):
                    class_wise_predict.append(int((all_preds == i).sum()))
                self.class_wise_predict.append(class_wise_predict)
            self.ensemble_acc.append(ens_acc)
            self.logger.info("{0:25} : {1:10f}".format("Ensemble acc", ens_acc))
            self.logger.info(
                "{0:25} : {1:10f}".format("Max acc", max(self.ensemble_acc))
            )
            self.check_and_save_weights(current_iter)
            if current_iter == self.pretrain_iter:
                self.pretrain_acc = ens_acc

            self.logger.info("--------- Acc on {:10} ----------".format(self.trgt))
            for i in range(len(self.src)):
                acc = f1_score(all_labels, domain_wise_preds[i], average="micro")
                self.logger.info(
                    "{0:25} : {1:10f}".format(self.trgt + " on " + self.src[i], acc)
                )
                self.domain_acc[self.src[i]].append(acc)

    def evaluation(self, current_iter, weight):
        self.logger.info(">>>>>>>>> Evaluate at {:6} <<<<<<<<<".format(current_iter))
        self.network.eval()
        self.val(current_iter, weight)
        self.save_metrics()
        self.network.train()

    def save_metrics(self):
        filename = os.path.join(self.log_dir, self.filename + ".json")

        metrics = {}
        metrics["max_acc"] = max(self.ensemble_acc)
        metrics["pretrain_acc"] = self.pretrain_acc
        metrics["ensemble_acc"] = self.ensemble_acc
        metrics["pseudo_label_metric"] = self.pseudo_label
        metrics["class_wise_num"] = self.class_wise_num
        metrics["class_wise_predict"] = self.class_wise_predict
        metrics["domain_acc"] = self.domain_acc

        with open(filename, "w") as f:
            json.dump(metrics, f)

        self.logger.info("Metrics saved at " + filename + "\n")
