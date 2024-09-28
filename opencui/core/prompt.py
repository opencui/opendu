from abc import ABC, abstractmethod
from typing import Callable

from opencui import Task, RauConfig


# Let's setup instruction builder
class InstructBuilder(ABC):
    @abstractmethod
    def build(self, **kwargs):
        pass

#
# For each class of problem, we might have many different prompt template, assumes the same set of variables.
# eventually, this will be a global manager, so that we can specify prompt template (instruction builder)
# by it's label.
#
class PromptManager(ABC):
    @abstractmethod
    def __getitem__(self, label) -> InstructBuilder:
        pass

    def get_builder(self, task: Task):
        match task:
            case Task.SKILL:
                return self[RauConfig.get().skill_prompt]
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