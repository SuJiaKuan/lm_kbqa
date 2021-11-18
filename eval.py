import argparse

import numpy as np
import torch
from tqdm import tqdm

from kbqa.tokenizer import load_tokenizer
from kbqa.dataset import load_datasets
from kbqa.model import load_model
from kbqa.const import DATASET
from kbqa.const import MODEL_ARCHITECTURE
from kbqa.const import TASK
from kbqa.const import SEQUENCE_LABEL
from kbqa.config import SEQUENCE_LABELING_ID2LABEL
from kbqa.config import SEQUENCE_LABELING_LABEL2ID


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for kbqa evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Model checkpoint to load",
    )
    parser.add_argument(
        "data",
        type=str,
        help="Path to data directory",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=DATASET.SIMPLE_QUESTIONS,
        help="Dataset to load",
        choices=[
            DATASET.SIMPLE_QUESTIONS,
        ],
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=MODEL_ARCHITECTURE.BERT,
        help="Model architecture to load",
        choices=[
            MODEL_ARCHITECTURE.BERT,
        ]
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Path to output directory",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="__project__cache__",
        help="Path to cache directory",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable cache",
    )

    args = parser.parse_args()

    return args


def eval_accuracy(test_dataloader, model, device):
    answer_id = SEQUENCE_LABELING_LABEL2ID[SEQUENCE_LABEL.ANSWER]
    correct_cnt = 0
    total_cnt = 0

    model_ = model.to(device)

    print("Calculate accuracy...")
    for encodings in tqdm(test_dataloader):
        encodings_ = {k: v.to(device) for k, v in encodings.items()}
        predictions = model_(**encodings_)

        gt_labels = encodings["labels"].numpy()
        pred_labels = np.argmax(
            predictions.logits.detach().cpu().numpy(),
            axis=-1,
        )

        answer_indices = gt_labels == answer_id
        correctnesses = pred_labels[answer_indices] == answer_id

        correct_cnt += np.sum(correctnesses)
        total_cnt += len(correctnesses)

    return correct_cnt, total_cnt


def main(args):
    tokenizer = load_tokenizer(args.model, args.checkpoint)
    test_dataset = load_datasets(
        args.dataset,
        args.data,
        ["test"],
        tokenizer,
        args.cache,
        not args.no_cache,
    )[0]
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
    )

    model = load_model(
        TASK.SEQUENCE_LABELING,
        args.checkpoint,
        SEQUENCE_LABELING_ID2LABEL,
        SEQUENCE_LABELING_LABEL2ID,
    )

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    correct_cnt, total_cnt = eval_accuracy(test_dataloader, model, device)
    print("Accuracy: {}% ({} / {})".format(
        100 * (correct_cnt / total_cnt),
        correct_cnt,
        total_cnt,
    ))


if __name__ == "__main__":
    main(parse_args())
