#!/usr/bin/env python3
import tempfile
import chromadb
import hashlib
import json
import yaml
from typing import Dict, List, TypedDict, Union
from chromadb.api.models.Collection import Collection


class LoadDataError(Exception):
    pass


class ValidationDataError(Exception):
    pass


class FunctionParametersProperties(TypedDict):
    _type: str
    description: str


class FunctionParameters:
    _type: str
    properties: Dict[str, FunctionParametersProperties]


class Function:
    name: str
    description: str
    parameters: FunctionParameters


class ExemplarObject(TypedDict):
    _type: str
    utterance: str


class Exemplar:
    funcId: str
    exemplars: List[ExemplarObject]


class Converter:

    def __init__(self, path=tempfile.mkdtemp()) -> None:
        self.chroma_client = chromadb.PersistentClient(path=path)

    def _addIndex(self,
                  c: Collection,
                  func: List[Function] = [],
                  exemplars: List[Exemplar] = []):
        if len(func) > 0:
            docs = [f.description for f in func]
            ids = [hashlib.sha256(f.encode('UTF-8')).hexdigest() for f in docs]
            metadatas = []
            for f in func:
                _p = {}
                for p in f.parameters.properties.values():
                    _p[p.get("_type")] = p.get("description")
                metadatas.append(_p)

            c.add(ids=ids, metadatas=metadatas, documents=docs)

        if len(exemplars) > 0:
            docs = [e.funcId for e in exemplars]
            ids = [hashlib.sha256(f.encode('UTF-8')).hexdigest() for f in docs]
            metadatas = []
            for e in exemplars:
                metadatas.append(e.exemplars)

            c.add(ids=ids, metadatas=metadatas, documents=docs)

    @staticmethod
    def parseSpecs(data: dict) -> List[Function]:
        resutl = []
        for _, v in data.get("paths", {}).items():
            for _, _v in v.items():
                f = Function()
                f.name = _v.get("operationId", "")
                description = _v.get("description", "")
                if description == "":
                    description = _v.get("summary", "")
                f.description = description

                p = FunctionParameters()
                p._type = "object"
                p.properties = {}

                for _p in _v.get("parameters", []):
                    p.properties[_p.get("name", "")] = {
                        "_type": _p.get("type", ""),
                        "description": _p.get("description", "")
                    }
                resutl.append(f)
                f.parameters = p

        return resutl

    def addSpecs(self, name: str = "specs", data: Union[str, dict] = ""):
        if len(data) == 0:
            return

        if type(data) == str:
            try:
                if data[0] == "{":
                    _data = json.loads(data)
                else:
                    _data = yaml.safe_load(data)
            except:
                raise LoadDataError
        elif type(data) == dict:
            _data = data
        else:
            raise LoadDataError

        func = self.parseSpecs(_data)

        collection = self.chroma_client.get_or_create_collection(name=name)
        self._addIndex(collection, func=func)

    def addExemplars(self, name: str = "exemplars", data: List[Exemplar] = []):
        collection = self.chroma_client.get_or_create_collection(name=name)
        self._addIndex(collection, exemplars=data)
