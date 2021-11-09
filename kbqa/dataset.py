import operator
import re

from fuzzysearch import Match
from fuzzysearch import find_near_matches
from thefuzz import process

from kbqa.util import word_tokenize_with_indices


def _match_best_question_entities(question, names, max_l_dist=3):
    indexed_word_tokens = word_tokenize_with_indices(question)
    word_tokens = [w[0] for w in indexed_word_tokens]

    clean_names = [n.strip() for n in names]
    name_candidates = process.extractBests(question, clean_names)

    if not name_candidates:
        return word_tokens, []

    best_score = max(name_candidates, key=operator.itemgetter(1))[1]
    name_candidates = list(filter(
        lambda c: c[1] == best_score,
        name_candidates,
    ))

    entity_indices_lst = []
    for name, _ in name_candidates:
        matches = [
            Match(start=m.start(), end=m.end(), dist=0, matched=name)
            for m in re.finditer(name, question)
        ]
        if not matches:
            matches = find_near_matches(name, question, max_l_dist=max_l_dist)
            matches = list(filter(lambda m: m.matched, matches))

        if not matches:
            continue

        min_dist = min(matches, key=lambda m: m.dist).dist
        matches = list(filter(lambda m: m.dist == min_dist, matches))

        for match in matches:
            entity_indices = []
            for token_index, (_, start, end) in enumerate(indexed_word_tokens):
                if not(end <= match.start or start >= match.end):
                    entity_indices.append(token_index)
            entity_indices_lst.append(entity_indices)

    entity_indices_lst = sorted(
        entity_indices_lst,
        key=operator.length_hint,
    )
    filterd_entity_indices_lst = []
    for idx, entity_indices in enumerate(entity_indices_lst):
        has_intersection = any([
            bool(set(entity_indices).intersection(m))
            for m in entity_indices_lst[idx+1:]
        ])
        if not has_intersection:
            filterd_entity_indices_lst.append(entity_indices)

    return word_tokens, filterd_entity_indices_lst
