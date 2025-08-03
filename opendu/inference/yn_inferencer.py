# Copyright 2024, OpenCUI
# Licensed under the Apache License, Version 2.0.
# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel
from opendu import FrameSchema
from opendu.core.annotation import SlotSchema
from opendu.core.config import Task
from opendu.core.prompt import PromptManager
from opendu.inference.decoder import Decoder, OutputExpectation


#
# There are two levels of the APIs here: a
# 1. assuming the entity recognizer is external,
# 2. assuming the entity recognizer is internel.
# The difference will be whether we include candidates in the API.
# We will be focusing on the #1 as #2 can be easily implemented with #1.
#
# There are many possible solutions for this:
# For example, question/answering based: https://arxiv.org/pdf/2406.08848
# For structure extraction bases: nuextract
# https://arxiv.org/pdf/2403.17536v1

class YesNoQuestion(BaseModel):
    text: str
    dialogActType: Optional[str] = None
    frame: Optional[str] = None
    slot: Optional[str] = None


# The modes that we will support.
YesNoResult = Enum("YesNoResult", ["Affirmative", "Negative", "Indifferent", "Irrelevant"])

#
# The slot extractor takes task description as context, and slot schema, candidate values and
# generate the json representation of values.
#
class YesNoInferencer(ABC):
    def __init__(self, retriever = None):
        self.generator = Decoder.get()
        self.retriever = retriever
        self.yni_prompt = PromptManager.get_builder(Task.Yni)
        self.yni_results = ["Affirmative", "Negative", "Indifferent", "Irrelevant"]
        
    def decide(self, utterance:str, question:YesNoQuestion) -> YesNoResult:
        # For now, we ignore the examples, so we do not need retriever yet.
        input_dict = {"response": utterance, "question": question.text, "examples": []}
        input_prompt = self.yni_prompt(input_dict)

        raw_output = self.generator.generate(input_prompt, OutputExpectation(choices=self.yni_results))
        print(raw_output)
        return raw_output[0].outputs[0].text


if __name__ == "__main__":

    yni = YesNoInferencer()

    text = "I can handle that."
    question = YesNoQuestion(text="Are you sure you can hanbdle that?")
    
    outputs = yni.decide(text, question)
    print(outputs)


    text = "I really doubt it."
    outputs = yni.decide(text, question)
    print(outputs)

    text = "I am not sure."
    outputs = yni.decide(text, question)
    print(outputs)


    text = "I like to open a new account."
    outputs = yni.decide(text, question)
    print(outputs)
