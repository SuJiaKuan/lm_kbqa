import numpy as np
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification


def main():
    tokenizer_checkpoint = "bert-base-cased"
    model_checkpoint = "./results/checkpoint-500"
    sentence = \
        "@Feabries I have never worked on Google, which is located on America."

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_checkpoint)
    encodings = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    model = BertForTokenClassification.from_pretrained(model_checkpoint)
    predictions = model(**encodings)

    tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])[1:-1]
    tag_ids = np.argmax(predictions.logits.detach().numpy()[0], axis=-1)[1:-1]

    print(sentence)
    print("====")
    for token, tag_id in zip(tokens, tag_ids):
        tag = model.config.id2label[tag_id]
        if tag != "O":
            print("{}\t{}".format(token, tag))
    print("====")


if __name__ == "__main__":
    main()
