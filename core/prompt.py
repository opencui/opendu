#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
import abc
from abc import ABC
import logging
from dataclasses import dataclass, field

from langchain.schema import BaseRetriever
from llama_index.schema import TextNode

from core.annotation import ModuleSchema
from core.retriever import HybridRetriever
from pybars import Compiler
import random


#
# This assumes the dataset has skills, skill_descriptions, slots, slot_descriptions
# Then user utterance as input, and output.
#
@dataclass
class Prompt:
    __metaclass__ = abc.ABCMeta
    extra_tokens: list[str] = field(default_factory=list)

    @abc.abstractmethod
    def __call__(self, item: dict[str, str]) -> str:
        # Expecting: utterance, [skills, slots, examples]
        return


#
# We will use eos: </s> automatically in both train and decode. Prompt can decide whether
# and how they want to use bos: <s>.
# We need to have two path: one for training (we need extra_tokens) and one for decoding.
# In LUG, we call prompt needed by embedding instruction, as they are static. Templated prompt
# needed by generation will be called as prompt.
#


# Simple prompt only has utterance.
class SimplePrompt(Prompt, ABC):
    def __init__(self, source: str):
        self.template = Compiler().compile(source)
        self.extra_tokens = []

    def __call__(self, item: dict[str, str]) -> str:
        return self.template(item)


class ObjectLister:
    def __init__(
            self,
            item_header=None,
            header_delim: str = "\n",
            item_delim: str = "\n\n",
            block_header: str = "",
            block_tail: str = "",
            with_index: bool = True):
        self.item_header = item_header
        self.header_delim = header_delim
        self.item_delim = item_delim
        self.block_header = block_header
        self.block_tail = block_tail
        self.with_index = with_index

    def __call__(self, this, options, items):
        result = []
        result.append(self.block_header)
        for index, thing in enumerate(items):
            if index != 0:
                result.append(self.item_delim)
            if self.item_header:
                if self.with_index:
                    result.append(f'{self.item_header} {index}) {self.header_delim}')
                else:
                    result.append(f'{self.item_header} {self.header_delim}')
            result.extend(options['fn'](thing))
        result.append(self.block_tail)
        return result


#
# Assume the source have examples, slots, skills, values as well as utterance.
# This prompt template is designed to address the function representation
class FullPrompt(Prompt, ABC):
    def __init__(
            self,
            source: str,
            module: ModuleSchema,
            retriever: BaseRetriever = None,
            topk: int = 3,
            train_mode: bool = False,
            extra_tokens: list[str] = []):
        self.template = Compiler().compile(source)
        self.retriever = retriever
        self.skills = module.skills
        self.slots = module.slots
        self.topk = topk
        self.train_mode = train_mode
        self.extra_tokens = extra_tokens
        self.helpers = {
            'list_examples': ObjectLister(item_header="### Examples"),
            'list_skills': ObjectLister(item_header="### Functions", item_delim=",", block_header="[", block_tail="]"),
            'list_slots': ObjectLister(item_header=None, item_delim=",", block_header="[", block_tail="]"),
            'list_values': ObjectLister(item_header=None, item_delim=",", block_header="[", block_tail="]")
        }
        self.partials = {}

    def dedup(self, old_results: list[TextNode]):
        new_results = []
        intents = set()
        for item in old_results:
            intent = item.metadata["target_intent"]
            if intent not in intents:
                intents.add(intent)
                new_item = {"template": item.text, "owner": item.metadata["target_full"]}
                new_results.append(new_item)
            if len(new_results) >= self.topk:
                break
        random.shuffle(new_results)
        return new_results[:self.topk]

    def __call__(self, item: dict[str, any]) -> str:
        # First we need to create the example.
        if self.retriever:
            resultsWithScore = self.retriever.retrieve(item["utterance"])
            results = map(lambda x: x.node, resultsWithScore)
            if self.train_mode and 'id' in item.keys():
                results = filter(lambda x: x.id_ != item['id'], results)
            item["examples"] = self.dedup(results)

        item["skills"] = self.skills
        item["slots"] = self.slots

        return self.template(item, helpers=self.helpers, partials=self.partials)




#
# LugPrompts assumes the following prompt template in pybars depending on the following information:
# skills: List[SkillSpec]
# slots: List[SlotSpec]
# exemplars: List[Exemplar]
# values: ?
#
SKillPrompts = {
    "simple_prompt":
        "<s> Convert the input text to structured representation. ### Input: {{utterance}} ### Output:",
    "full_simple_prompt_txt00":
        """
        <s> Given the input sentence, construct a function representation of this sentence, including the function name,
        parameters, and their corresponding values. This function representation should describe the target sentence 
        accurately.  
         
        The function must be one of the following 
        {{#list_skills skills}} {{name}} {{/list_skills}}
        .
        
        For each parameter with its value mentioned in the sentence, enclose the parameter and its corresponding values in
         brackets. The parameters must be one of the following:
        {{#list_slots slots}} {{name}} {{/list_slots}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
    "full_exampled_prompt":
        """
        <s> Given the input sentence, construct a function representation of this sentence, including the function name,
         parameters, and their corresponding values. This function representation should describe the target sentence 
         accurately and the function must be one of the following 
        {{#list_skills skills}} {{name}} {{/list_skills}}
        .
        For each parameter with its value mentioned in the sentence, enclose the parameter and its corresponding values in
         brackets. The parameters must be one of the following:
        {{#list_slots slots}} {{name}} {{/list_slots}}
        The order your list the parameters within the function must follow the order listed above. 
        
        Here are a couple of examples.
        {{#list_examples examples}} Sentence: {{template}} \n Output: {{owner}} \n {{/list_examples}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
    "exampled_prompt_for_skill00":
        """<s> 
        Given an input sentence, a set of functions with names and their descriptions, as well as some example templates
         of how to express these functions in natural language text, the goal is to determine the function implied by 
        the input sentence. The selected function should accurately describe the target sentence, and it should 
        be one of the following functions:

        {{#list_skills skills}} {{owner}} : {{description}} {{/list_skills}} . 
         
        Here are a couple of example templates:
        {{#list_examples examples}} ### Input template: {{template}} \n ### Output: {{owner}} \n {{/list_examples}}
        
        ### Input sentence: \n
        {{utterance}}
        ### Output: \n
        """,

    "exampled_prompt_for_skill01":
        """<s> 
        Given an input sentence, a set of functions with names and their descriptions, as well as some example templates
         of how to express these functions in natural language text, the goal is to determine the function implied by 
        the input sentence. The selected function should accurately describe the target sentence:

        {{#list_skills skills}} {{owner}} : {{description}} {{/list_skills}} . 
         
        Here are a couple of example templates:
        {{#list_examples examples}} ### Input template: {{template}} \n ### Output: {{owner}} \n {{/list_examples}}
        
        ### Input sentence: \n
        {{utterance}}
        ### Output: \n
        """,
}

AllSlotPrompts = {
    "exampled":
        """
        <s> Given the input sentence, specification of a function, including name and description of the function and 
        its parameters, the task is to extract the values for these parameters from the input sentence (if mentioned) 
        and return the extracted parameters and their values in JSON format. \n
       
        Here is the function: \n
        {{#list_skills skills}} {{name}} : {{description}} {{/list_skills}}
        .
        
        For each parameter with its value mentioned in the sentence, extract the mentioned value for these parameters.
        The parameters must be one of the following:
        {{#list_slots slots}} {{name}}: {{description}} {{/list_slots}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
}

# For the slots of enum type, we used different prompt in order to improve the
EnumPrompts = {
        "default" : """"
        <s> Given an input sentence, extract the value for parameter {{name}}, {{description}}, from the input sentence.
        
        Here are possible values for this parameter:
        {{#list_values values}} value {{/list_values}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
}

OneSlotPrompts = {
    "default":
        """
        <s> Given an input sentence, extract the value for parameter {{name}}, {{description}}, from the input sentence.
        
        Here are possible values for this parameter:
        {{#list_values values}} value {{/list_values}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
}
