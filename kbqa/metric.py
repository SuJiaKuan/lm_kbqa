import numpy as np

from datasets import load_metric


def compute_seq_labeling_metrics(model_outputs):
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
