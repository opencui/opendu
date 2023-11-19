import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from core.annotation import build_nodes_from_exemplar_store
from core.embedding import EmbeddingStore
from core.prompt import SkillPrompts, SlotPrompts
from core.retriever import build_desc_index, load_context_retrievers, build_nodes_from_skills, create_index
from finetune.commons import build_dataset_index, build_nodes_from_dataset
from finetune.generation import SkillTrainConverter, OneSlotTrainConverter, ConvertedFactory

#
# Converter is a lower level component of converter. This directly use the model.
# This assumes there are fine-tune model already, but use the same client code (albeit different code path)
#
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    from finetune.sgd import SGD
    from converter.lug_config import LugConfig

    LugConfig.embedding_device = "cuda"
    factories = [
        SGD("/home/sean/src/dstc8-schema-guided-dialogue/")]

    # For now, just use the fix path.
    output = "./test"

    # Save the things to disk first.
    desc_nodes = []
    exemplar_nodes = []
    for factory in factories:
        build_nodes_from_skills(factory.tag, factory.module_schema.skills, desc_nodes)
        build_nodes_from_dataset(factory.tag, factory.build("test"), exemplar_nodes)

    create_index(f"{output}/index", "exemplar", exemplar_nodes)
    create_index(f"{output}/index", "desc", desc_nodes)

    schemas = [factory.schema for factory in factories]
    context_retriever = load_context_retrievers(factory.schema, f"{output}/index")
    skill_converter0 = SkillTrainConverter(context_retriever, SkillPrompts["specs_exampled"])
    skill_converter1 = SkillTrainConverter(context_retriever, SkillPrompts["specs_only"])
    slot_converter = OneSlotTrainConverter(factory.schema, SlotPrompts["basic"])



    converted_factories.append(ConvertedFactory(factory, [skill_converter0, skill_converter1, slot_converter]))


    output = "./index/viggo/"
    prompt = get_prompt(viggo, output)

    convert = Converter("./output/503B_FT_lr1e-5_ep5_top1_2023-09-26/checkpoint-3190/")

    dataset = Viggo("full").build("test")
    counts = [0, 0]
    marker = "### Output:"
    for item in dataset:
        sequences = convert(item, prompt)
        counts[0] += 1
        seq = sequences[0]
        text = seq['generated_text']
        idx = text.index(marker)
        result = text[idx+len(marker):].strip()
        item_id = item["id"]
        result = get_func(result)
        target = get_func(item['target_full'])
        if result == target:
            counts[1] += 1
        else:
            print(f"{result} != {target}\n")
    print(counts)