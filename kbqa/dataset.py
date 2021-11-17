import operator
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from fuzzysearch import Match
from fuzzysearch import find_near_matches
from thefuzz import process
from transformers import BertTokenizerFast

from kbqa.io import CacheManager
from kbqa.kg import FreebaseKnowledgeGraph
from kbqa.util import word_tokenize
from kbqa.util import word_tokenize_with_indices
from kbqa.util import is_ascii
from kbqa.const import DATASET
from kbqa.const import MODEL_ARCHITECTURE
from kbqa.const import SEQUENCE_LABEL
from kbqa.config import SIMPLE_QUESTIONS_CONFIG
from kbqa.config import SEQUENCE_LABELING_LABEL2ID


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
            for m in re.finditer(re.escape(name), question)
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


def load_datasets(
    dataset,
    data_dir,
    splits,
    model_arch,
    checkpoint,
    cache_dir,
    cache_enabled,
):
    datasets = []

    cm = CacheManager(cache_dir, enabled=cache_enabled)

    if model_arch == MODEL_ARCHITECTURE.BERT:
        tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    else:
        raise ValueError(
            "Non-suppoted model architecture: {}".format(model_arch),
        )

    if dataset == DATASET.SIMPLE_QUESTIONS:
        kg_filepath = os.path.join(
            data_dir,
            "freebase-subsets",
            "freebase-FB2M.txt",
        )
        names_filepath = os.path.join(
            data_dir,
            "freebase_names",
            "names.trimmed.2M.txt",
        )
        kg = FreebaseKnowledgeGraph(kg_filepath, names_filepath, cm)

        for split in splits:
            dataset_filepath = os.path.join(
                data_dir,
                "annotated_fb_data_{}.txt".format(split),
            )
            datasets.append(SimpleQuestionsDataset(
                dataset_filepath,
                kg,
                tokenizer,
                cm,
            ))
    else:
        raise ValueError("Non-suppoted dataset: {}".format(dataset))

    return datasets


class SimpleQuestionsDataset(torch.utils.data.Dataset):

    def __init__(self, filepath, kg, tokenizer, cm):
        cache_name_prefix = "{}|{}|{}".format(
            filepath,
            kg.kg_filepath,
            kg.names_filepath,
        )
        self._raw_examples = cm.load(
            "{}|raw_examples".format(cache_name_prefix),
            self._load,
            filepath,
            kg,
        )
        self._encodings = cm.load(
            "{}|encodings".format(cache_name_prefix),
            self._encode,
            self._raw_examples,
            tokenizer,
        )

    def __getitem__(self, idx):
        return self._encodings[idx]

    def __len__(self):
        return len(self._encodings)

    def _load(self, filepath, kg):
        filepath_obj = Path(filepath)

        raw_text = filepath_obj.read_text().strip()
        lines = re.split(r"\n", raw_text)

        raw_examples = []
        for line in lines:
            subj_uri, relation_uri, obj_uri, question = line.split("\t")

            topic_entity = kg.get_entity(subj_uri)
            answer_entity = kg.get_entity(obj_uri)
            triplets = kg.get_triplets(subj_uri)

            word_tokens, entity_indices_lst = _match_best_question_entities(
                question,
                topic_entity.names,
            )
            triplets = kg.get_triplets(subj_uri)

            raw_examples.append({
                "question": question,
                "word_tokens": word_tokens,
                "topic_entity": topic_entity,
                "entity_indices_lst": entity_indices_lst,
                "answer_entity": answer_entity,
                "triplets": triplets,
            })

        return raw_examples

    def _choose_entity_name(self, names, en_first=True):
        if not names:
            return ""

        if en_first:
            en_names = list(filter(is_ascii, names))
            if en_names:
                return en_names[0]
            else:
                return names[0]
        else:
            return names[0]

    def _choose_triplets(self, triplets, answer_entity, max_num):
        answer_triplets = []
        normal_triplets = []

        for triplet in triplets:
            if triplet.obj_entity.uri == answer_entity.uri:
                answer_triplets.append((triplet, True))
            else:
                normal_triplets.append((triplet, False))

        num_normals = min(max_num - len(answer_triplets), len(normal_triplets))
        chosen_triplets = \
            random.sample(normal_triplets, k=num_normals) + answer_triplets
        random.shuffle(chosen_triplets)

        return chosen_triplets

    def _process_triplets(self, triplets):
        processed_triplets = []

        for triplet, is_answer in triplets:
            relation_uri = triplet.relation.uri
            relation_name = relation_uri.split("/")[-1].replace("_", " ")
            relation_word_tokens = word_tokenize(relation_name)

            obj_names = triplet.obj_entity.names
            # XXX (SuJiaKuan):
            # We specify en_first as True because we assume the language is
            # English. You may need specify it as False when handling other
            # languages.
            obj_name = self._choose_entity_name(obj_names, en_first=True)
            obj_word_tokens = word_tokenize(obj_name)

            processed_triplets.append((
                relation_word_tokens,
                obj_word_tokens,
                is_answer,
            ))

        return processed_triplets

    def _equip_triplets(self, word_tokens, entity_indices_lst, triplets):
        equipped_word_tokens = []
        word_labels = []
        word_token_structures = []

        last_indices = [indices[-1] for indices in entity_indices_lst]
        inner_block_idx = 1
        num_total_added_tokens = 0
        for idx, word_token in enumerate(word_tokens):
            equipped_word_tokens.append(word_token)
            word_token_structures.append({
                "level": 0,
                "block": 0,
                "parent": [],
                "children": [],
            })

            word_labels.append(SEQUENCE_LABEL.DISABLED)

            if idx in last_indices:
                parent_indices = entity_indices_lst[last_indices.index(idx)]
                parent_indices = [
                    i + num_total_added_tokens
                    for i in parent_indices
                ]

                for relation_word_tokens, obj_word_tokens, is_answer \
                        in triplets:
                    num_added_tokens = \
                        len(relation_word_tokens) + len(obj_word_tokens)
                    child_indices = list(range(
                        len(equipped_word_tokens),
                        len(equipped_word_tokens) + num_added_tokens,
                    ))
                    num_total_added_tokens += num_added_tokens

                    equipped_word_tokens += relation_word_tokens
                    equipped_word_tokens += obj_word_tokens
                    for added_idx in range(num_added_tokens):
                        word_token_structures.append({
                            "level": 1,
                            "block": inner_block_idx,
                            "parent": parent_indices,
                            "children": [],
                        })

                        if added_idx == 0:
                            label = \
                                SEQUENCE_LABEL.ANSWER \
                                if is_answer \
                                else SEQUENCE_LABEL.NON_ANSWER
                        else:
                            label = SEQUENCE_LABEL.DISABLED
                        word_labels.append(label)

                    for parent_idx in parent_indices:
                        word_token_structures[parent_idx]["children"].append(
                            child_indices,
                        )

                    inner_block_idx += 1

        return equipped_word_tokens, word_labels, word_token_structures

    def _encode_model_token_ids(self, word_tokens, tokenizer):
        tokenizer_encoded = tokenizer.encode_plus(
            word_tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
        )
        model_token_ids = tokenizer_encoded["input_ids"]
        offset_mapping = tokenizer_encoded["offset_mapping"]

        return model_token_ids, offset_mapping

    def _structurize_model_tokens(
        self,
        word_tokens,
        word_labels,
        word_token_structures,
        model_token_ids,
        offset_mapping,
        tokenizer,
    ):
        model_to_word_indices = []
        model_to_word_idx = -1
        word_to_model_indices_lst = []
        word_to_model_indices = []
        for model_token_idx, (model_token_id, offset) \
                in enumerate(zip(model_token_ids, offset_mapping)):
            if offset[0] == 0 and offset[1] == 0:
                model_to_word_indices.append(-1)
            else:
                if offset[0] == 0:
                    model_to_word_idx += 1
                model_to_word_indices.append(model_to_word_idx)

                word_to_model_indices.append(model_token_idx)
                if model_token_idx == (len(model_token_ids) - 1) \
                        or offset_mapping[model_token_idx + 1][0] == 0:
                    word_to_model_indices_lst.append(word_to_model_indices)
                    word_to_model_indices = []

        # Handle the case that model tokens are truncated by tokenizer.
        for _ in range(len(word_tokens) - len(word_to_model_indices_lst)):
            word_to_model_indices_lst.append([])

        model_labels = []
        model_token_structures = []
        for model_token_idx, (model_token_id, model_to_word_idx) in \
                enumerate(zip(model_token_ids, model_to_word_indices)):
            if model_token_id in \
                    [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                model_token_structures.append({
                    "level": 0,
                    "block": 0,
                    "parent": [],
                    "children": [],
                })
                model_labels.append(SEQUENCE_LABEL.DISABLED)
            elif model_token_id == tokenizer.pad_token_id:
                model_token_structures.append({
                    "level": -1,
                    "block": -1,
                    "parent": [],
                    "children": [],
                })
                model_labels.append(SEQUENCE_LABEL.DISABLED)
            else:
                word_token_structure = word_token_structures[model_to_word_idx]

                parent_indices = []
                for parent_idx in word_token_structure["parent"]:
                    parent_indices += word_to_model_indices_lst[parent_idx]

                child_indices_lst = []
                for child_indices in word_token_structure["children"]:
                    child_indices_ = []
                    for child_idx in child_indices:
                        child_indices_ += word_to_model_indices_lst[child_idx]
                    child_indices_lst.append(child_indices_)

                model_token_structures.append({
                    "level": word_token_structure["level"],
                    "block": word_token_structure["block"],
                    "parent": parent_indices,
                    "children": child_indices_lst,
                })

                leading_model_token_idx =\
                    word_to_model_indices_lst[model_to_word_idx][0]
                if model_token_idx == leading_model_token_idx:
                    word_label = word_labels[model_to_word_idx]
                    model_labels.append(word_label)
                else:
                    model_labels.append(SEQUENCE_LABEL.DISABLED)

        return model_labels, model_token_structures

    def _encode_labels(self, model_labels):
        labels = []
        for model_label in model_labels:
            if model_label == SEQUENCE_LABEL.DISABLED:
                labels.append(-100)
            else:
                labels.append(SEQUENCE_LABELING_LABEL2ID[model_label])

        return labels

    def _encode_position_ids(self, model_token_structures):
        position_ids = []
        outer_idx = 0
        inner_idx = 0
        for model_token_idx, model_token_structure \
                in enumerate(model_token_structures):
            if model_token_structure["level"] in [-1, 0]:
                position_ids.append(outer_idx)
                outer_idx += 1
            else:
                is_block_start = \
                    (model_token_idx == 0) \
                    or (model_token_structures[model_token_idx - 1]["block"]
                        != model_token_structure["block"])
                inner_idx = outer_idx if is_block_start else inner_idx + 1
                position_ids.append(inner_idx)

        return position_ids

    def _encode_token_type_ids(self, model_token_structures):
        return [0] * len(model_token_structures)

    def _encode_attention_mask(self, model_token_structures):
        attention_mask = []
        levels = np.array([s["level"] for s in model_token_structures])
        blocks = np.array([s["block"] for s in model_token_structures])

        for model_token_structure in model_token_structures:
            if model_token_structure["level"] == -1:
                attention_line = np.zeros(len(model_token_structures))
            else:
                attention_line = np.logical_and(
                    levels == model_token_structure["level"],
                    blocks == model_token_structure["block"],
                )
                attention_line = attention_line.astype(np.int)
                for parent_idx in model_token_structure["parent"]:
                    attention_line[parent_idx] = 1
                for child_indices in model_token_structure["children"]:
                    for child_idx in child_indices:
                        attention_line[child_idx] = 1

            attention_mask.append(attention_line.tolist())

        return attention_mask

    def _encode(self, raw_examples, tokenizer):
        encodings = []

        for raw_example in raw_examples:
            triplets = self._choose_triplets(
                raw_example["triplets"],
                raw_example["answer_entity"],
                SIMPLE_QUESTIONS_CONFIG["MAX_TRIPLETS"]
            )
            triplets = self._process_triplets(triplets)

            word_tokens = raw_example["word_tokens"]
            entity_indices_lst = raw_example["entity_indices_lst"]

            word_tokens, word_labels, word_token_structures = \
                self._equip_triplets(word_tokens, entity_indices_lst, triplets)

            model_token_ids, offset_mapping = self._encode_model_token_ids(
                word_tokens,
                tokenizer,
            )

            model_labels, model_token_structures = \
                self._structurize_model_tokens(
                    word_tokens,
                    word_labels,
                    word_token_structures,
                    model_token_ids,
                    offset_mapping,
                    tokenizer,
                )

            labels = self._encode_labels(model_labels)
            position_ids = self._encode_position_ids(model_token_structures)
            token_type_ids = self._encode_token_type_ids(
                model_token_structures,
            )
            attention_mask = self._encode_attention_mask(
                model_token_structures,
            )

            encoding = {
                "input_ids": model_token_ids,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            encoding = {
                key: torch.tensor(val)
                for key, val in encoding.items()
            }

            encodings.append(encoding)

        return encodings
