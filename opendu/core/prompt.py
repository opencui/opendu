#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
import json
from abc import ABC
from opendu.core.config import RauConfig
from enum import Enum
from jinja2 import Environment, FileSystemLoader


# We only work with well-defined task.
class Task(Enum):
    IdBc = "id_bc",
    SfSs = "sf_ss",
    Yni = "yni"


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
    def get_builder(task: str, input_mode: bool = True):
        print(f"**************************** {task}")
        return PromptManager.get(RauConfig.get().prompt[task], input_mode)

    @staticmethod
    def get_task_label(task: str):
        return RauConfig.get().prompt[task].split(".")[0]


if __name__ == "__main__":
    print(Task.IdBc.value[0])

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
    utterance = "I can not do it on Wednesday"
    schema_str = """{
        "LocalDate": {
            "description": "type used to capture the date without time zone attached.",
            "candidates": ["Tuesday", "July 4th"]
        },
        "book_dental_visit": {
            "schema": {
                "start_date":  {
                    "type" : "LocalDate",
                    "description": "the date when the reservation starts"
                },
                "number_of_reservations": {
                    "type": "integer",
                    "description": "how many times the reservation has to repeat."
                }
            },
            "description": "the semantic frame needed for booking the repeated reservations"
        }
    }"""

    examples_str = """[
        {
            "input": "I like to book a 5 times reservation starting on tuesday.",
            "output": {
                "start_data": {
                    "=": "tuesday"
                },
                "number_of_reservations": {
                    "=": 5
                }
            }
        }
    ]"""

    schema = json.loads(schema_str)
    examples = json.loads(examples_str)

    x = {
        "skill": "book_dental_visit",
        "utterance": utterance,
        "schema": schema,
        "examples": examples
    }

    print(PromptManager.get("sf_se_default")(x))
    #print(examples)