from core.annotation import SemanticStructure, Exemplar
from finetune.commons import SkillSpec


#
# Here are the work that client have to do.
# For now, assume single module. We can worry about multiple module down the road.
# 1. use description and exemplars to find built the prompt for the skill prompt.
# 2. use skill and recognizer to build prompt for slot.
# 3. stitch together the result.
#
class Converter:
    def __init__(self, retrievers, schema, recognizers):
        self.schema = schema
        self.retrievers = retrievers
        self.recognizers = recognizers
        self.llm = None

    def to_structure(self, text: str) -> SemanticStructure:
        desc_nodes = self.retrievers[0].retrieve(text)
        exemplar_nodes = self.retrievers[1].retrieve(text)

        selected_skills = get_skill_infos(self.schema, desc_nodes + exemplar_nodes)
        selected_exemplars = get_examplars(exemplar_nodes)



        # Now we need to create prompt for the function first.
        func_name = None
        # Then we need to create the prompt for the parameters.

        slot_values = None
        return SemanticStructure(name = func_name, arguments = slot_values)

    def to_text(self, struct:SemanticStructure) -> str:
        llm = self.llm
        # To be defined.
        return None


def get_skill_infos(skills, nodes) -> list[SkillSpec]:
    funcset = { item.node.meta["target_intent"] for item in nodes}
    return [skills[func] for func in funcset]


def get_examplars(nodes) -> list[Exemplar]:
    return [Exemplar(owner=item.node.meta["target_intent"]) for item in nodes]
