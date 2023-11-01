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

from core.annotation import ModuleSpec
from finetune.datasets import SGDSkills
from finetune.commons import DatasetFactory, DatasetFactoryWrapper
from core.embedding import EmbeddingStore
from core.retriever import HybridRetriever, build_nodes_from_skills, build_nodes_from_dataset, create_index
from pybars import Compiler
import random

from finetune.datasets.viggo import Viggo


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
            module: ModuleSpec,
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
            'list_examples': ObjectLister(item_header="### Example"),
            'list_skills': ObjectLister(item_header="### Functions", item_delim=",", block_header="[", block_tail="]"),
            'list_slots': ObjectLister(item_header=None, item_delim=",", block_header="[", block_tail="]"),
            'list_values': ObjectLister()
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

    @classmethod
    def build_index(cls, dsc: DatasetFactory, output: str = "./output/"):
        desc_nodes = build_nodes_from_skills(dsc.domain.skills)
        exemplar_nodes = build_nodes_from_dataset(dsc.build("train"))

        create_index(output, "desc", desc_nodes, EmbeddingStore.for_description())
        create_index(output, "exemplars", exemplar_nodes, EmbeddingStore.for_exemplar())


#
# LugPrompts assumes the following prompt template in pybars depending on the following information:
# skills: List[SkillSpec]
# slots: List[SlotSpec]
# exemplars: List[Exemplar]
# values: ?
#
Prompts = {
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
    "exampled_prompt_for_slots":
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
    "exampled_prompt_for_skill":
        """
        <s> Given the input sentence, construct a function representation of this sentence, including the function name,
         parameters, and their corresponding values. This function representation should describe the target sentence 
         accurately and the function must be one of the following 
        {{#list_skills skills}} {{name}} {{/list_skills}}
        .    
        Here are a couple of example templates.
        {{#list_examples examples}} Sentence: {{template}} \n Output: {{owner}} \n {{/list_examples}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
}


#
# Assume the node id_ is the same as dataset id.
#
def get_prompt(dsc: DatasetFactory, index_path: str) -> Prompt:
    retriever = HybridRetriever(index_path)
    return FullPrompt(Prompts["full_exampled_prompt"], dsc.domain, retriever=retriever)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    viggo = SGDSkills()
    output = "./index/viggo/"
    #print(compute_k(viggo.build("validation"), output, retriever))

    prompt = get_prompt(viggo,  output)
    dsc = DatasetFactoryWrapper(Viggo("full"), prompt)
    dataset = dsc.build("train")
    for item in dataset:
        print(item)
        print("\n\n")
