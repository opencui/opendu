#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
import html
from pybars import Compiler
from abc import ABC, abstractmethod
from opendu.core.config import RauConfig
from enum import Enum
from jinja2 import Environment, FileSystemLoader, Template


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

#
# Assume the source have examples, slots, skills, values as well as utterance.
# This prompt template is designed to address template for full skill.
#
class PromptBuilder:
    def __init__(self, template):
        self.template = template

    # Assume __call__ takes object, but build take scatter parts.
    def __call__(self, kwargs) -> str:
        return self.template.render(**kwargs)


#
# For each class of problem, we might have many different prompt template, assumes the same set of variables.
# eventually, this will be a global manager, so that we can specify prompt template (instruction builder)
# by it's label.
#
class PromptManager(ABC):
    @staticmethod
    def get(label, input_mode: bool = True):
        env = Environment(loader=FileSystemLoader("./opendu/core/templates/"))
        task = "input" if input_mode else "output"
        label = label.replace('-', '_')
        return PromptBuilder(env.get_template(f"{label}.{task}"))

    @staticmethod
    def get_builder(task: Task, input_mode: bool = True):
        print(f"**************************** {task}")
        return PromptManager.get(RauConfig.get().prompt[task], input_mode)


    @staticmethod
    def get_task_label(task: Task):
        return RauConfig.get().prompt[task].split(".")[0]


# We should be able to switch to different manager later.
PromptManager = PromptManager


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

    print(PromptManager.get("yni_default")(x))

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

    print(PromptManager.get("skill_knn_structural")(x))