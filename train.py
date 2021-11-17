import argparse

from transformers import TrainingArguments
from transformers import Trainer

from kbqa.dataset import load_datasets
from kbqa.model import load_model
from kbqa.metric import compute_seq_labeling_metrics
from kbqa.const import DATASET
from kbqa.const import MODEL_ARCHITECTURE
from kbqa.const import TASK
from kbqa.config import SEQUENCE_LABELING_ID2LABEL
from kbqa.config import SEQUENCE_LABELING_LABEL2ID


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
        "--epochs",
        type=int,
        default=3,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Strenght of weight decay",
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
    train_dataset = datasets[0]
    val_dataset = datasets[1]

    model = load_model(
        TASK.SEQUENCE_LABELING,
        args.checkpoint,
        SEQUENCE_LABELING_ID2LABEL,
        SEQUENCE_LABELING_LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_seq_labeling_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main(parse_args())
