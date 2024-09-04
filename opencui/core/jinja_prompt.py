import jinja2


#
# Assume the source have examples, slots, skills, values as well as utterance.
# This prompt template is designed to address template for full skill.
class JinjaPrompt:
    environment = jinja2.Environment()

    def __init__(self, source: str):
        self.template = JinjaPrompt.environment.from_string(source)

    def __call__(self, item: dict[str, any]) -> str:
        # First we need to create the example.
        return self.template.render(item)



# Each class prompt assumes the same set of variables.
class TypedPrompts:
    def __init__(self, **kvargs):
        self.prompt_map = {
            key: JinjaPrompt(source) for key, source in kvargs.items()
        }

    def __getitem__(self, item):
        return self.prompt_map[item]


#
# RauPrompts assumes the following prompt template in pybars depending on the following information:
# skills: List[SkillSpec]
# slots: List[SlotSpec]
# exemplars: List[Exemplar]
# values: ?
