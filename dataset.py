import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from aug_list import aug_list


def create_index(utility):
    sample_num = {}
    if utility.dataset == "domain-net":
        for dom in utility.domains:
            sample_num[dom] = [0] * utility.label_num
            with open(
                os.path.join(
                    utility.path,
                    utility.dataset,
                    "index_main",
                    "_".join([dom, "train.txt"]),
                ),
                "r",
            ) as f:
                image_index = []
                data = f.readlines()
                for line in data:
                    img_path, label = line.split(" ")
                    image_index.append(
                        [
                            os.path.join(
                                utility.path, utility.dataset, img_path.strip()
                            ),
                            label,
                        ]
                    )
                    sample_num[dom][int(label)] += 1
                np.save(
                    open(
                        os.path.join(utility.index_dir, "_".join([dom, "train.npy"])),
                        "wb",
                    ),
                    image_index,
                )
            with open(
                os.path.join(
                    utility.path,
                    utility.dataset,
                    "index_main",
                    "_".join([dom, "test.txt"]),
                ),
                "r",
            ) as f:
                image_index = []
                data = f.readlines()
                for line in data:
                    img_path, label = line.split(" ")
                    image_index.append(
                        [
                            os.path.join(
                                utility.path,
                                utility.dataset,
                                img_path.strip(),
                            ),
                            label,
                        ]
                    )
            np.save(
                open(
                    os.path.join(utility.index_dir, "_".join([dom, "test.npy"])),
                    "wb",
                ),
                image_index,
            )

    else:
        for dom in utility.domains:
            sample_num[dom] = [0] * utility.label_num
            dir = os.path.join(utility.path, utility.dataset, dom)
            classes = [x[0] for x in os.walk(dir)]
            classes = classes[1:]
            classes.sort()
            image_index = []
            for label, abs_dir in enumerate(classes):
                image_index.extend(
                    [
                        [os.path.abspath(os.path.join(abs_dir, img_path)), label]
                        for img_path in os.listdir(abs_dir)
                    ]
                )
                sample_num[dom][int(label)] = len(os.listdir(abs_dir))
            np.save(
                open(
                    os.path.join(utility.index_dir, "_".join([dom, "test.npy"])),
                    "wb",
                ),
                image_index,
            )
            np.save(
                open(
                    os.path.join(utility.index_dir, "_".join([dom, "train.npy"])),
                    "wb",
                ),
                image_index,
            )
    save_path = os.path.join(
        utility.exp, utility.dataset, "".join([utility.dataset, ".json"])
    )
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            json.dump(sample_num, f)


class ImageList(Dataset):
    def __init__(self, index_file_dir, aug, index, labels):
        self.index_list = np.load(index_file_dir, allow_pickle=True)
        if index != None:
            self.index_list = self.index_list[index]
        self.aug = aug
        self.aug_list_src = aug_list
        self.aug_list_trgt = aug_list
        if self.aug:
            if labels != None:
                self.image_list = [
                    [idx, x[0], int(labels[idx]), augment]
                    for idx, x in enumerate(self.index_list)
                    for augment in self.aug_list_trgt
                ]
            else:
                self.image_list = [
                    [idx, x[0], int(x[1]), augment]
                    for idx, x in enumerate(self.index_list)
                    for augment in self.aug_list_src
                ]
        else:
            self.image_list = [
                [idx, x[0], int(x[1]), aug_list[-1]]
                for idx, x in enumerate(self.index_list)
            ]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        index, img_path, label, augment = self.image_list[idx]
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        img = augment(img)
        return index, img, label
