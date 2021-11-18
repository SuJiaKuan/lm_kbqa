from transformers import BertTokenizerFast

from kbqa.const import MODEL_ARCHITECTURE


def load_tokenizer(model_arch, checkpoint):
    if model_arch == MODEL_ARCHITECTURE.BERT:
        tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    else:
        raise ValueError(
            "Non-suppoted model architecture: {}".format(model_arch),
        )

    return tokenizer
