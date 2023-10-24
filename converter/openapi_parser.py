#!/usr/bin/env python3

from abc import ABC
import json
import yaml
from typing import Dict, List, TypedDict, Union
from core.commons import SkillInfo, DatasetCreator, SlotInfo


#
# This is used to create dataset need for build index from OpenAPI specs.
#
class OpenAPIParser(DatasetCreator, ABC):

    def __init__(self, path) -> None:
        specs = json.load(open(path))
        skills = OpenAPIParser.parseSpecs(specs)

    @staticmethod
    def parseSpecs(data: dict) -> List[SkillInfo]:
        result = []
        for path, v in data.get("paths", {}).items():
            for op, _v in v.items():
                f = SkillInfo()
                f.name = _v.get("operationId", "")
                description = _v.get("description", "")
                if description == "":
                    description = _v.get("summary", "")
                f.description = description

                p = SlotInfo()
                p._type = "object"
                p.properties = {}

                for _p in _v.get("parameters", []):
                    p.properties[_p.get("name", "")] = {
                        "_type": _p.get("type", ""),
                        "description": _p.get("description", "")
                    }
                result.append(f)
                f.parameters = p

        return result

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
                raise RuntimeError("Can't load data")
        elif type(data) == dict:
            _data = data
        else:
            raise RuntimeError("Can't load data")

        func = self.parseSpecs(_data)


if __name__ == "__main__":
    s = OpenAPIParser("./converter/openapi_example.json")
    assert len(s) == 4
    assert s[0].name == "findPets"
    assert "Returns all pets from the system that" in s[0].description
    _tags = s[0].parameters.properties.get("tags", {})
    assert _tags.get("_type", "") == "array"
    assert _tags.get("description", "") == "tags to filter by"


