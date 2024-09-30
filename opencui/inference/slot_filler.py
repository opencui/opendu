# Copyright 2024, OpenCUI
# Licensed under the Apache License, Version 2.0.

import json
from abc import ABC, abstractmethod
from pydantic import BaseModel
from collections import defaultdict
from enum import Enum

from opencui import FrameSchema
from opencui.core.annotation import (CamelToSnake, DialogExpectation, Exemplar, OwnerMode, ExactMatcher)
from opencui.core.config import RauConfig
from opencui.core.prompt import (DescriptionPrompts, ExemplarPrompts)
from opencui.core.retriever import (ContextRetriever)
from opencui.inference.generator import GenerateMode

from opencui.utils.json_tools import parse_json_from_string

#
# There are two levels of the APIs here: a
# 1. assuming the entity recognizer is external,
# 2. assuming the entity recognizer is internel.
# The difference will be whether we include candidates in the API.
# We will be focusing on the #1 as #2 can be easily implemented with #1.
#

#
# The slot filler takes task description, and slot descriptions, candidate values and
# generate the json representation of values.
#
#
class SlotFiller(ABC):
    @abstractmethod
    def extract_values(self, text, expectations:FrameSchema, candidates: dict):
        pass

