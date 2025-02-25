#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
import html

from pybars import Compiler
from abc import ABC, abstractmethod
from typing import Callable

from opendu.core.config import RauConfig
from abc import ABC
from typing import Callable
from enum import Enum

from jinja2 import Environment, FileSystemLoader

# We only work with well-defined task.
class Task(Enum):
    SKILL = 1,
    SLOT = 2,
    YNI = 3,
    BOOL_VALUE = 4,
    HAS_MORE = 5,
    SKILL_DESC = 6

class IOMode(Enum):
    INPUT = "input",
    OUTPUT = "output"

# Let's setup instruction builder
class InstructBuilder(ABC):

    def __call__(self, **kwargs):
        return self.build(**kwargs)

    @abstractmethod
    def build(self, **kwargs):
        pass

#
# For each class of problem, we might have many different prompt template, assumes the same set of variables.
# eventually, this will be a global manager, so that we can specify prompt template (instruction builder)
# by it's label.
#
class PromptManager(ABC):
    def __getitem__(self, label) -> InstructBuilder:
        return self.get(label)

    @abstractmethod
    def get(self, label):
        pass

    def get_builder(self, task: Task, mode: IOMode = None):
        print(f"**************************** {task}")
        match task:
            case Task.SKILL:
                if mode is None:
                    return self[RauConfig.get().skill_prompt]
                elif mode == IOMode.INPUT:
                    return self[f"{RauConfig.get().skill_prompt}.input"]
                else:
                    return self[f"{RauConfig.get().skill_prompt}.output"]
            case Task.SKILL_DESC:
                return self[RauConfig.get().skill_desc_prompt]
            case Task.SLOT:
                return self[RauConfig.get().slot_prompt]
            case Task.YNI:
                return self[RauConfig.get().yni_prompt]
            case Task.BOOL_VALUE:
                return self[RauConfig.get().bool_prompt]

    def get_task_label(self, task: Task):
        match task:
            case Task.SKILL:
                return RauConfig.get().skill_prompt.split(".")[0]
            case Task.SKILL_DESC:
                return RauConfig.get().skill_desc_prompt.split(".")[0]
            case Task.SLOT:
                return RauConfig.get().slot_prompt.split(".")[0]
            case Task.YNI:
                return RauConfig.get().yni_prompt.split(".")[0]
            case Task.BOOL_VALUE:
                return RauConfig.get().bool_prompt.split(".")[0]


#
# Assume the source have examples, slots, skills, values as well as utterance.
# This prompt template is designed to address template for full skill.
#
class JinjaPromptBuilder(InstructBuilder, ABC):
    def __init__(self, label: str):
        env = Environment(loader=FileSystemLoader("opendu/core/templates"))
        self.template = env.get_template(label)

    # Assume __call__ takes object, but build take scatter parts.
    def __call__(self, kwargs) -> str:
        return self.build(**kwargs)

    def build(self, **kwargs) -> str:
        return self.template.render(**kwargs)


# Notice this manager does not need to
class JinjaPromptManager(PromptManager, ABC):
    def get(self, label):
        return JinjaPromptBuilder(label)


# We should be able to switch to different manager later.
promptManager1 = JinjaPromptManager()

#
# We will use eos: </s> automatically in both train and decode. Prompt can decide whether
# and how they want to use bos: <s>.
# We need to have two path: one for training (we need extra_tokens) and one for decoding.
# In LUG, we call prompt needed by embedding instruction, as they are static. Templated prompt
# needed by generation will be called as prompt.
#
class ObjectLister:
    def __init__(
        self,
        item_header=None,
        item_header_delim: str = "\n",
        item_delim: str = "\n",
        block_header: str = "",
        block_tail: str = "",
        with_index: bool = True,
    ):
        self.item_header = item_header
        self.item_header_delim = item_header_delim
        self.item_delim = item_delim
        self.block_header = block_header
        self.block_tail = block_tail
        self.with_index = with_index

    def __call__(self, this, options, items):
        result = []
        # If the item list is empty.
        if items is None or len(items) == 0:
            return result
        result.append(self.block_header)
        for index, thing in enumerate(items):
            if self.item_header:
                if self.with_index:
                    result.append(
                        f"{self.item_header} {index}) {self.item_header_delim}"
                    )
                else:
                    result.append(f"{self.item_header} {self.item_header_delim}")
            result.extend(options["fn"](thing))
            result.append(self.item_delim)
        result.append(self.block_tail)
        return result


#
# Assume the source have examples, slots, skills, values as well as utterance.
# This prompt template is designed to address template for full skill.
class PybarsPrompt(InstructBuilder):
    default_helpers = {
            "list_examples": ObjectLister(),
            "list_skills": ObjectLister(
                block_header="\nGiven the definition for the following functions:\n"
            ),
            "list_skill_names": ObjectLister(
                block_header="Classify the input into the following functions:\n"
            ),
            "list_slots": ObjectLister(
                item_header=None, item_delim=",", block_header="[", block_tail="]"
            ),
            "list_values": ObjectLister(
                item_header=None,
                item_delim=",",
                block_header="Given the candidate values: [",
                block_tail="null]",
            ),
        }

    def __init__(self, source: str, helpers = default_helpers):
        self.template = Compiler().compile(source)
        self.source = source
        self.extra_tokens = []
        self.helpers = helpers
        self.partials = {}

    def __call__(self, item: dict[str, any]) -> str:
        # First we need to create the example.
        return html.unescape(
            self.template(item, helpers=self.helpers, partials=self.partials)
        )

    def build(self, **kwargs):
        # First we need to create the example.
        return html.unescape(
            self.template(kwargs, helpers=self.helpers, partials=self.partials)
        )


class PybarsPromptManager(PromptManager):
    def __init__(self):
        self.collections = {
        "skill-desc-structural": PybarsPrompt(
            'Decide whether the input fit this function description: {{skill.description}}'
            'Input: {{utterance}} \n\n Decision:'
        ),
        "skill-knn-structural": PybarsPrompt(

            'Decide whether the following template means the same as:\n {{utterance}}.\n\n'
            '{{#list_examples examples}}Template: {{template}}\nDecision:{{label}}\n\n{{/list_examples}}'
            'Template: {{template}}\nDecision:'),
        "slot-qa-structural": PybarsPrompt(
            'Mark the value for {{name}} in utterance.\n\n'
            '{{#list_examples examples}}Utterance: {{utterance}}\nOutput:{{label}}\n\n{{/list_examples}}'
            'Utterance: {{utterance}}\nOutput:'),
        "yni-default": PybarsPrompt(
            'Decide whether the response is affirmative, negative, indifferent or irrelevant to this '
            'yes/no '
            'question:\n\n'
            '{{question}}\n'
            '{{#list_examples examples}}Response: {{response}}\nDecision:{{label}}\n\n{{/list_examples}}'
            'Response: {{response}}\nDecision:'),
        "bool-value-plain": PybarsPrompt('{{label}}'),

    }

    def get(self, label) -> InstructBuilder:
        return self.collections[label]


# We should be able to switch to different manager later.
promptManager0 = PybarsPromptManager()

#
# LugPrompts assumes the following prompt template in pybars depending on the following information:
# skills: List[SkillSpec]
# slots: List[SlotSpec]
# exemplars: List[Exemplar]
# values: ?
#
MulticlassSkillPrompts = {
    "specs_only": PybarsPrompt(
        "{{#list_skills skills}} {{name}}: {{description}} {{/list_skills}}\n"
        'Classify the input sentence: "{{utterance}}" as one of '
        '{{#list_skill_names skills}} "{{name}}" {{/list_skill_names}}.\n'
        'The prediction is:'
    ),
    "specs_exampled": PybarsPrompt(
        '{{#list_skills skills}} "{{name}}": {{description}} {{/list_skills}}\n'
        "{{#list_examples examples}}Input template: {{template}} means {{owner}}\n{{/list_examples}}\n"
        'Classify the input sentence: "{{utterance}}" as one of '
        '{{#list_skill_names skills}} "{{name}}" {{/list_skill_names}}.\n'
        "The prediction is:"
    ),
    "full": PybarsPrompt(
        '{{#list_skills skills}} "{{name}}": {{description}} {{/list_skills}}\n'
        "Classify the input as one of "
        '{{#list_skill_names skills}} "{{name}}" {{/list_skill_names}}.\n\n'
        '{{#list_examples examples}}Input: "{{template}}"\nOutput:"{{owner}}"\n{{/list_examples}}\n'
        'Input: "{{utterance}}"\nOutput:'
    ),
    "basic": PybarsPrompt(
        '{{#list_skill_names skills}} "{{name}}": {{description}} {{/list_skill_names}}\n'
        "Output null if input does not imply any one of them.\n\n"
        '{{#list_examples examples}}Input: "{{template}}"\nOutput:"{{owner}}"\n{{/list_examples}}\n'
        'Input: "{{utterance}}"\nOutput:'
    ),
}

BinarySkillPrompts = {
    "default": PybarsPrompt(
        'Determine whether the input means "{{skill.name}}": {{skill.description}}, output true or false.\n'
        '{{#list_examples examples}}Input: [{{template}}] means "{{target}}"? Output: {{decision}}{{/list_examples}}'
        'Input: [{{utterance}}] means "{{skill.name}}"? Output: '
    ),
    "natural": PybarsPrompt(
        'Determine whether the input means "{{skill.name}}": {{skill.description}}, output true or false.\n'
        "{{#list_examples examples}}"
        'Question: Does "{{template}}" mean "{{target}}"? Answer: {{decision}}'
        "{{/list_examples}}"
        'Question: Does "{{utterance}}" mean "{{skill.name}}"? Answer:'
    ),
}


DescriptionPrompts = {
    "default": PybarsPrompt(
        'Is it true that "{{utterance}}" fits the description "{{skill.description}}"?'
    ),
    "structural": PybarsPrompt(
        'Decide whether the input fit this function description: {{skill.description}}'
        'Input: {{utterance}} \n\n Decision:'
    ),
    "struct-short": PybarsPrompt(
        'Given the function description:\n "{{skill.description}}" \n'
        'and the sentence: {{utterance}} \n'
        'Is it true that the sentence fits the function description?'
    ),
    "token": PybarsPrompt(
        '<utterance> {{utterance}} </utterance> <func> {{skill.name}} : {{skill.description}} </func> <description>'
    ),
    "struct-token": PybarsPrompt(
        'Given the function '
        '<func_name> {{skill.name}} </func_name> with its description: <func_desc> {{skill.description}} </func_desc>, '
        'decide whether <utterance> {{utterance}} </utterance> means this function.'
        'The answer is'
    ),
    "struct-token1": PybarsPrompt(
        'Given:\n'
        'the utterance: <utterance> {{utterance}} </utterance>\n'  
        'the function description: <func_desc> {{skill.description}} </func_desc>\n'
        'Is it true that the utterance fit the function description?'
        '<desc> The answer is '
    ),
    "struct-token2": PybarsPrompt(
        'Given:\n'
        'the utterance: <utterance> {{utterance}} </utterance>\n'
        'the function description: <func_desc> {{skill.description}} </func_desc>\n'
        'Is it true that the utterance fit the function description?'
        '<desc> '
    ),
}


# This should have the same exact key as the above prompt dictionary.
ExemplarPrompts = {
    "default": PybarsPrompt(
        'Is it true that "{{utterance}}" means the same as the example: "{{template}}"?'),
    "structural": PybarsPrompt(
        'Decide whether the following template means the same as:\n {{utterance}}.\n\n'
        '{{#list_examples examples}}Template: {{template}}\nDecision:{{label}}\n\n{{/list_examples}}'
        'Template: {{template}}\nDecision:'),
}

# For the slots of enum type, we used different prompt in order to improve the
# Candidates should be , separated string for now.
ExtractiveSlotPrompts = {
    "default": PybarsPrompt(
        '{{#list_values values}} {{value}} {{/list_values}}\n'
        'The value for {{name}} from "{{utterance}}" is:'),
    "structural": PybarsPrompt(
        'Mark the value for {{name}} in utterance.\n\n'
        '{{#list_examples examples}}Utterance: {{utterance}}\nOutput:{{label}}\n\n{{/list_examples}}'
        'Utterance: {{utterance}}\nOutput:'),
}

YniPrompts = {
    "default": PybarsPrompt(
        'Decide whether the response is affirmative, negative, indifferent or irrelevant to this '
        'yes/no '
        'question:\n\n'
        '{{question}}\n'
        '{{#list_examples examples}}Response: {{response}}\nDecision:{{label}}\n\n{{/list_examples}}'
        'Response: {{response}}\nDecision:')
}


NliPrompts = {
    "boolq": PybarsPrompt(
        'Given the premise: "{{premise}}", the hypothesis: "{{hypothesis}}" is its ')
}


BoolPrompts = {
    "default": PybarsPrompt('{{label}}</s>'),
    "plain": PybarsPrompt('{{label}}'),
    "truefalse": PybarsPrompt('{{label}}</true_false_response></s>'),
    "yesno": PybarsPrompt('{{label}}</yes_no_response></s>')
}


if __name__ == "__main__":

    examples = [
        {"response": "April 2st", "label": "related"},
        {"response": "April 3st", "label": "unrelated"}
    ]
    x = {
        "question": "what day is tomorrow?",
        "response": "April 1st",
        "label": "related",
        "examples": examples
    }

    print(YniPrompts["default"](x))

    print(promptManager1["yn-default"](x))

    examples = [
        {"template": "April 2st", "label": "related"},
        {"template": "April 3st", "label": "unrelated"}
    ]
    x = {
        "template": "what day is tomorrow?",
        "utterance": "April 1st",
        "label": "related",
        "examples": examples
    }
    print(ExemplarPrompts["structural"](x))
    print(promptManager0["skill-knn-structural"](x))

     # print(promptManager.get_builder(Task.SKILL))