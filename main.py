import os, time, torch, random, argparse, cv2
import numpy as np
from trainer import Trainer
from utility import Utility


def main():
    parser = argparse.ArgumentParser(description="S3DA-LC")
    parser.add_argument("--dataset", type=str, default="office-31")
    parser.add_argument("--task", type=str, default="DW_A")
    parser.add_argument("--exp", type=str, default="exp")
    parser.add_argument("--path", type=str, default="/path/to/data/")
    parser.add_argument("--load_iter", type=int, default=0)
    parser.add_argument("--worker", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no", type=str, default="1")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--w_k", type=int, default=1, choices=[0, 1])
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--UTF", type=float, default=1.5, choices=[-1, 1.5, 3.0])
    args = parser.parse_args()

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    Trainer(args, Utility(args)).train()


if __name__ == "__main__":
    main()
