"""Classes and functions for knowledge graphs"""

import re
from abc import ABC
from abc import abstractmethod
from pathlib import Path

from tqdm import tqdm

from kbqa.exception import InvalidKnowledgeGraphKey
from kbqa.exception import KnowledgeGraphKeyError


FB_URI_PREFIX = "www.freebase.com"
FB_ID_PREFIX = "fb"


class Entity(object):

    def __init__(self, uri, id_, names):
        self._uri = uri
        self._id = id_
        self._names = names

    def __repr__(self):
        return "Entity(\'{}\', \'{}\', {})".format(
            self._uri,
            self._id,
            self._names,
        )

    __str__ = __repr__

    @property
    def uri(self):
        return self._uri

    @property
    def id(self):
        return self._id

    @property
    def names(self):
        return self._names


class Relation(object):

    def __init__(self, uri, id_):
        self._uri = uri
        self._id = id_

    def __repr__(self):
        return "Relation(\'{}\', \'{}\')".format(self._uri, self._id)

    __str__ = __repr__

    @property
    def uri(self):
        return self._uri

    @property
    def id(self):
        return self._id


class Triplet(object):

    def __init__(self, subj_entity, relation, obj_entity):
        self._subj_entity = subj_entity
        self._relation = relation
        self._obj_entity = obj_entity

    def __repr__(self):
        return "Triplet({}, {}, {})".format(
            repr(self._subj_entity),
            repr(self._relation),
            repr(self._obj_entity),
        )

    def __str__(self):
        return "(\n subject: {},\n relation: {},\n object: {}\n)".format(
            str(self._subj_entity),
            str(self._relation),
            str(self._obj_entity),
        )

    @property
    def subj_entity(self):
        return self._subj_entity

    @property
    def relation(self):
        return self._relation

    @property
    def obj_entity(self):
        return self._obj_entity


class KnowledgeGraph(ABC):

    @abstractmethod
    def get_entity(self, key):
        pass

    @abstractmethod
    def get_triplets(self, key):
        pass


class FreebaseKnowledgeGraph(KnowledgeGraph):

    def __init__(self, kg_filepath, names_filepath, cm):
        self._kg_filepath = kg_filepath
        self._names_filepath = names_filepath

        print("Loading entity names from '{}'".format(names_filepath))
        self._entity_names_mapping = cm.load(
            names_filepath,
            self._load_entity_names,
            names_filepath,
        )
        print("Loading knowledge graph from '{}'".format(kg_filepath))
        self._kg = cm.load(
            kg_filepath,
            self._load_kg,
            kg_filepath,
            self._entity_names_mapping,
        )

    @property
    def kg_filepath(self):
        return self._kg_filepath

    @property
    def names_filepath(self):
        return self._names_filepath

    def _load_kg(self, filepath, entity_names_mapping):
        filepath_obj = Path(filepath)

        kg = {}
        raw_text = filepath_obj.read_text().strip()
        lines = re.split(r"\n", raw_text)
        for line in tqdm(lines):
            subj_uri, relation_uri, obj_uris = line.split("\t")
            obj_uris = obj_uris.split(" ")

            if subj_uri not in kg:
                kg[subj_uri] = set()

            for obj_uri in obj_uris:
                kg[subj_uri].add((relation_uri, obj_uri))

        return kg

    def _load_entity_names(self, filepath):
        filepath_obj = Path(filepath)

        raw_text = filepath_obj.read_text().strip()
        lines = re.split(r"\n", raw_text)

        entity_name_mapping = {}
        for line in tqdm(lines):
            fid, _, name = line.split("\t")
            if fid not in entity_name_mapping:
                entity_name_mapping[fid] = [name]
            else:
                entity_name_mapping[fid].append(name)

        return entity_name_mapping

    def _format_key(self, key):
        if key.startswith(FB_URI_PREFIX):
            uri_key = key
            id_key = self.uri_to_id(key)
        elif key.startswith(FB_ID_PREFIX):
            uri_key = self.id_to_uri(key)
            id_key = key
        else:
            raise InvalidKnowledgeGraphKey("Invalid key: {}".format(key))

        return uri_key, id_key

    def uri_to_id(self, uri):
        tokens = uri.split("/")[1:]

        return "{}:{}".format(FB_ID_PREFIX, ".".join(tokens))

    def id_to_uri(self, id_):
        tokens = id_.split(":")[1].split(".")

        return "{}/{}".format(FB_URI_PREFIX, "/".join(tokens))

    def get_entity(self, key, should_exist=False):
        entity_uri, entity_id = self._format_key(key)

        if should_exist and entity_uri not in self._kg:
            raise KnowledgeGraphKeyError(
                "Key not in knowledge graph: {}".format(key),
            )

        return Entity(
            entity_uri,
            entity_id,
            self._entity_names_mapping.get(entity_id, []),
        )

    def get_triplets(self, key):
        triplets = []

        subj_entity = self.get_entity(key)
        subj_uri, _ = self._format_key(key)
        for relation_uri, obj_uri in self._kg.get(subj_uri, []):
            relation = Relation(relation_uri, self.uri_to_id(relation_uri))
            obj_entity = self.get_entity(obj_uri)
            triplets.append(Triplet(subj_entity, relation, obj_entity))

        return triplets
