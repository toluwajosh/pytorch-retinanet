from tqdm import tqdm
import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import (
    CocoDataset,
    CSVDataset,
    collater,
    Resizer,
    AspectRatioBasedSampler,
    Augmenter,
    Normalizer,
)
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split(".")[0] == "1"

print("CUDA available: {}".format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Simple training script for training a RetinaNet network."
    )

    parser.add_argument("--dataset", help="Dataset type, must be one of csv or coco.")
    parser.add_argument("--model", default=None, help="Path to trained model")
    parser.add_argument("--coco_path", help="Path to COCO directory")
    parser.add_argument(
        "--csv_train", help="Path to file containing training annotations (see readme)"
    )
    parser.add_argument(
        "--csv_classes", help="Path to file containing class list (see readme)"
    )
    parser.add_argument(
        "--csv_val",
        help="Path to file containing validation annotations (optional, see readme)",
    )

    parser.add_argument(
        "--depth",
        help="Resnet depth, must be one of 18, 34, 50, 101, 152",
        type=int,
        default=50,
    )
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=100)
    parser.add_argument(
        "--result_dir",
        default="results",
        help="Path to store training results",
        type=str,
    )
    parser.add_argument(
        "--batch_num", default=8, help="Number of samples in a batch", type=int
    )

    parser = parser.parse_args(args)

    print(parser)

    # parameters
    BATCH_SIZE = parser.batch_num
    IMAGE_MIN_SIDE = 1440
    IMAGE_MAX_SIDE = 2560

    # Create the data loaders
    if parser.dataset == "coco":

        if parser.coco_path is None:
            raise ValueError("Must provide --coco_path when training on COCO,")
        # TODO: parameterize arguments for Resizer, and other transform functions
        # resizer: min_side=608, max_side=1024
        dataset_train = CocoDataset(
            parser.coco_path,
            # set_name="train2017",
            set_name="train_images_full",
            transform=transforms.Compose(
                [Normalizer(), Augmenter(), Resizer(passthrough=True),]
            ),
        )
        dataset_val = CocoDataset(
            parser.coco_path,
            # set_name="val2017",
            set_name="val_images_full",
            transform=transforms.Compose([Normalizer(), Resizer(passthrough=True),]),
        )

    elif parser.dataset == "csv":

        if parser.csv_train is None:
            raise ValueError("Must provide --csv_train when training on COCO,")

        if parser.csv_classes is None:
            raise ValueError("Must provide --csv_classes when training on COCO,")

        dataset_train = CSVDataset(
            train_file=parser.csv_train,
            class_list=parser.csv_classes,
            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
        )

        if parser.csv_val is None:
            dataset_val = None
            print("No validation annotations provided.")
        else:
            dataset_val = CSVDataset(
                train_file=parser.csv_val,
                class_list=parser.csv_classes,
                transform=transforms.Compose([Normalizer(), Resizer()]),
            )

    else:
        raise ValueError("Dataset type not understood (must be csv or coco), exiting.")

    sampler = AspectRatioBasedSampler(
        dataset_train, batch_size=BATCH_SIZE, drop_last=False
    )
    dataloader_train = DataLoader(
        dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler
    )

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            dataset_val, batch_size=BATCH_SIZE, drop_last=False
        )
        dataloader_val = DataLoader(
            dataset_val, num_workers=16, collate_fn=collater, batch_sampler=sampler_val
        )

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(
            num_classes=dataset_train.num_classes(), pretrained=True
        )
    elif parser.depth == 34:
        retinanet = model.resnet34(
            num_classes=dataset_train.num_classes(), pretrained=True
        )
    elif parser.depth == 50:
        retinanet = model.resnet50(
            num_classes=dataset_train.num_classes(), pretrained=True
        )
    elif parser.depth == 101:
        retinanet = model.resnet101(
            num_classes=dataset_train.num_classes(), pretrained=True
        )
    elif parser.depth == 152:
        retinanet = model.resnet152(
            num_classes=dataset_train.num_classes(), pretrained=True
        )
    else:
        raise ValueError("Unsupported model depth, must be one of 18, 34, 50, 101, 152")

    if parser.model:
        retinanet = torch.load(parser.model)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True
    )

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print("Num training images: {}".format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        p_bar = tqdm(dataloader_train)
        for iter_num, data in enumerate(p_bar):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet(
                        [data["img"].cuda().float(), data["annot"]]
                    )
                else:
                    classification_loss, regression_loss = retinanet(
                        [data["img"].float(), data["annot"]]
                    )

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                mean_loss = np.mean(loss_hist)
                p_bar.set_description(
                    f"Epoch: {epoch_num} | Iteration: {iter_num} | "
                    f"Class loss: {float(classification_loss.item()):.5f} | "
                    f"Regr loss: {float(regression_loss.item()):.5f} | "
                    f"Running loss: {mean_loss:.5f}"
                )

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == "coco":

            print("Evaluating dataset")

            coco_eval.evaluate_coco(
                dataset_val, retinanet, result_dir=parser.result_dir
            )

        elif parser.dataset == "csv" and parser.csv_val is not None:

            print("Evaluating dataset")

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        # TODO: Fix string formating mix (adopt homogeneous format)
        torch.save(
            retinanet.module,
            f"{parser.result_dir}/"
            + "{}_retinanet_{}.pt".format(parser.dataset, epoch_num),
        )

    retinanet.eval()

    torch.save(retinanet, "model_final.pt")


if __name__ == "__main__":
    main()
