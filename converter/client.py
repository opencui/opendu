

from converter.lugconfig import LugConfig
from core.annotation import SemanticStructure
from core.retriever import get_retrievers


#
# Here are the work that client have to do.
# For now, assume single module. We can worry about multiple module down the road.
# 1. use description and exemplars to find built the prompt for the skill prompt.
# 2. use skill and recognizer to build prompt for slot.
# 3. stitch together the result.
#

class Client:
    def __init__(self, index_path, schema, recoginizer):
        self.retrievers = get_retrievers(index_path)

    def convert(self, text: str) -> SemanticStructure:
        desc_nodes = self.retrievers[0].retrieve(text)
        exemplar_nodes = self.retrievers[1].retrieve(text)



        return None

    def convert(self, struct:SemanticStructure) -> str:
        return None
