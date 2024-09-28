#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
from abc import ABC
from typing import Callable

from jinja2 import Environment, FileSystemLoader
from opencui.core.prompt import InstructBuilder, PromptManager

#
# Assume the source have examples, slots, skills, values as well as utterance.
# This prompt template is designed to address template for full skill.
#
class JinjaPromptBuilder(InstructBuilder, ABC):
    def __init__(self, label: str):
        env = Environment(loader=FileSystemLoader("opencui/core/templates"))
        self.template = env.get_template(label)


    def __call__(self, **kwargs) -> str:
        # First we need to create the example.
        return self.build(**kwargs)

    def build(self, **kwargs) -> str:
        return self.template.render(**kwargs)


# Notice this manager does not need to
class JingaPromptManager(PromptManager, ABC):
    def __getitem__(self, label):
        return JinjaPromptBuilder(label)


jinjaPromptManager = JingaPromptManager()


if __name__ == "__main__":
    examples = [
        {"input": "April 2st", "outputs": ["related", "but"] },
        {"input": "April 3st", "outputs": ["unrelated"] }
    ]

    skills = [
        {"name": "get_movie_ticket", "description": "sell user movie tickets based on their preference like showtime, etc"},
        {"name": "get_weather_info", "description": "retrieve weather information for the location user specified."}
    ]

    prompt_builder = jinjaPromptManager["id_mc_full.input"]
    print(prompt_builder.build(examples=examples, skills=skills, utterance="I miss home."))