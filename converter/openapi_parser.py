#!/usr/bin/env python3

from abc import ABC
import json
import yaml
from typing import Dict, List, TypedDict, Union
from core.commons import SkillInfo, DatasetCreator, SlotInfo, DomainInfo


#
# This is used to create dataset need for build index from OpenAPI specs.
#
class OpenAPIParser(DatasetCreator, ABC):
    # TODO (this is not fully tested yet, do not use, leave this here as reminder that we need to work on this soon).
    def __init__(self, specs) -> None:
        self.exemplars = []

        skills = {}
        slots = {}
        for path, v in specs.get("paths", {}).items():
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
                skills.append(f)
                f.parameters = p

        self.domain = DomainInfo(skills, slots)


if __name__ == "__main__":
    s = OpenAPIParser("./converter/openapi_example.json")



