import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_metric
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split


class WNUTDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self._encodings = encodings
        self._labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self._encodings.items()
        }
        item["labels"] = torch.tensor(self._labels[idx])

        return item

    def __len__(self):
        return len(self._labels)


def read_wnut(filepath):
    filepath_obj = Path(filepath)

    raw_text = filepath_obj.read_text().strip()
    raw_docs = re.split(r"\n\t?\n", raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split("\n"):
            token, tag = line.split("\t")
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offsets in zip(labels, encodings.offset_mapping):
        # Create an empty list of -100.
        doc_enc_labels = np.ones(len(doc_offsets), dtype=int) * -100
        lst_offsets = np.array(doc_offsets)

        # Set labels whose first offset position is 0 and the second is not 0.
        doc_enc_labels[(lst_offsets[:, 0] == 0) & (lst_offsets[:, 1] != 0)] = \
            doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def compute_metrics(model_outputs):
    metric = load_metric("seqeval")

    predictions, labels = model_outputs
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [
        [pred for pred, label in zip(prediction_seq, label_seq)
            if label != -100]
        for prediction_seq, label_seq in zip(predictions, labels)
    ]
    true_labels = [
        [label for pred, label in zip(prediction_seq, label_seq)
            if label != -100]
        for prediction_seq, label_seq in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    data_filepath = "data/wnut17/wnut17train.conll"
    checkpoint = "bert-base-cased"

    texts, tags = read_wnut(data_filepath)

    train_texts, val_texts, train_tags, val_tags = train_test_split(
        texts,
        tags,
        test_size=.2,
    )
    print(len(train_texts), len(val_texts))
    print(texts[0][10:17], tags[0][10:17], sep='\n')

    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: tag_id for tag_id, tag in enumerate(unique_tags)}
    id2tag = {tag_id: tag for tag, tag_id in tag2id.items()}
    print(tag2id)
    print(id2tag)

    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    train_encodings = tokenizer(
        train_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
    )
    val_encodings = tokenizer(
        val_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
    )

    train_labels = encode_tags(train_tags, train_encodings, tag2id)
    val_labels = encode_tags(val_tags, val_encodings, tag2id)
    '''
    idx = 100
    print(train_encodings["input_ids"][idx])
    print(tokenizer.convert_ids_to_tokens(train_encodings["input_ids"][idx]))
    print(train_encodings["offset_mapping"][idx])
    print(train_labels[idx])
    print(train_tags[idx])
    '''

    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")
    train_dataset = WNUTDataset(train_encodings, train_labels)
    val_dataset = WNUTDataset(val_encodings, val_labels)

    model = BertForTokenClassification.from_pretrained(
        checkpoint,
        num_labels=len(unique_tags),
        id2label=id2tag,
        label2id=tag2id,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
