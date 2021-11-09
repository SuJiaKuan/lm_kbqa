import nltk


def word_tokenize_with_indices(text):
    tokens = nltk.word_tokenize(text)

    indexed_word_tokens = []
    start = 0
    for token in tokens:
        start = text.find(token, start)
        end = start + len(token)
        indexed_word_tokens.append((token, start, end))
        start = end

    return indexed_word_tokens
