#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
import html
from pybars import Compiler
from abc import ABC, abstractmethod
from opendu.core.config import RauConfig
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
        env = Environment(loader=FileSystemLoader("./opendu/core/templates/"))
        self.template = env.get_template(label)

    # Assume __call__ takes object, but build take scatter parts.
    def __call__(self, kwargs) -> str:
        return self.build(**kwargs)

    def build(self, **kwargs) -> str:
        return self.template.render(**kwargs)


# Notice this manager does not need to
class JinjaPromptManager(PromptManager, ABC):
    def get(self, label, task="input"):
        label = label.replace('-', '_')
        return JinjaPromptBuilder(f"{label}.{task}")


# We should be able to switch to different manager later.
promptManager = JinjaPromptManager()


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

    print(promptManager["yn_default"](x))

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

    print(promptManager["id_knn_structural"](x))