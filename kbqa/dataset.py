import operator

from fuzzysearch import find_near_matches
from thefuzz import process

from kbqa.util import word_tokenize_with_indices


def _match_best_question_entities(question, names, max_l_dist=3):
    indexed_word_tokens = word_tokenize_with_indices(question)
    word_tokens = [w[0] for w in indexed_word_tokens]

    clean_names = [n.strip() for n in names]
    name_candidates = process.extractBests(question, clean_names)

    if not name_candidates:
        return indexed_word_tokens, []

    best_score = max(name_candidates, key=operator.itemgetter(1))[1]
    name_candidates = list(filter(
        lambda c: c[1] == best_score,
        name_candidates,
    ))

    matched_indices_lst = []
    for name, _ in name_candidates:
        matches = find_near_matches(name, question, max_l_dist=max_l_dist)

        if not matches:
            continue

        min_dist = min(matches, key=lambda m: m.dist).dist
        matches = list(filter(lambda m: m.dist == min_dist, matches))

        for match in matches:
            token_indices = []
            for token_index, (_, start, end) in enumerate(indexed_word_tokens):
                if not(end <= match.start or start >= match.end):
                    token_indices.append(token_index)
            matched_indices_lst.append(token_indices)

    matched_indices_lst = sorted(
        matched_indices_lst,
        key=operator.length_hint,
    )
    filterd_matched_indices_lst = []
    for idx, matched_indices in enumerate(matched_indices_lst):
        has_intersection = any([
            bool(set(matched_indices).intersection(m))
            for m in matched_indices_lst[idx+1:]
        ])
        if not has_intersection:
            filterd_matched_indices_lst.append(matched_indices)

    return word_tokens, filterd_matched_indices_lst
