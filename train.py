import argparse

from kbqa.dataset import load_datasets
from kbqa.model import load_model
from kbqa.const import DATASET
from kbqa.const import MODEL_ARCHITECTURE
from kbqa.const import TASK
from kbqa.config import SEQUENCE_LABEL_ID2TAG
from kbqa.config import SEQUENCE_LABEL_TAG2ID


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for kbqa training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "-c",
        "--checkpoint",
        type=str,
        default="bert-base-cased",
        help="Model checkpoint to load",
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


def main(args):
    datasets = load_datasets(
        args.dataset,
        args.data,
        ["train", "valid"],
        args.model,
        args.checkpoint,
        args.cache,
        not args.no_cache,
    )

    model = load_model(
        TASK.SEQUENCE_LABELING,
        args.checkpoint,
        SEQUENCE_LABEL_ID2TAG,
        SEQUENCE_LABEL_TAG2ID,
    )


if __name__ == "__main__":
    main(parse_args())
