"""Classes and functions for Neural Network models"""

from transformers import BertForTokenClassification

from kbqa.const import TASK


def load_model(task, checkpoint, id2label, tag2label):
    if task == TASK.SEQUENCE_LABELING:
        model = BertForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=tag2label,
        )
    else:
        raise ValueError("Non-suppoted task: {}".format(task))

    return model
