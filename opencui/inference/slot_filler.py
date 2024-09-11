# Copyright 2024, OpenCUI
# Licensed under the Apache License, Version 2.0.

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum

from opencui.core.annotation import (CamelToSnake, DialogExpectation, Exemplar, OwnerMode, ExactMatcher)
from opencui.core.config import RauConfig
from opencui.core.pybars_prompt import (DescriptionPrompts, ExemplarPrompts)
from opencui.core.retriever import (ContextRetriever)
from opencui.inference.generator import GenerateMode

from opencui.utils.json_tools import parse_json_from_string

# The modes that we will support.
class SlotFiller(ABC):
    @abstractmethod
    def extract_values(self, text, expectations=None, debug=False):
        pass

    @abstractmethod
    def grade(self, text, owner, owner_mode, counts_dict):
        pass
