import json
from opencui import EmbeddingStore, ConvertedFactory, build_dataset_index, JsonDatasetFactory, \
    RauConfig, DescExemplarConverter, InstanceMode, MultiClassSkillConverter, TrainPhase1Converter
from opencui.core.retriever import build_desc_index, load_context_retrievers, ContextRetriever
from opencui.core.prompt import promptManager, Task


#
# Before we build dataset for skills, we need to handle the following steps:
# 1. extract schema.
# 2. extract exemplars: both for skill and slots.
# 3. extract entities.
# 4. build index.
# The separate indexing for desc and exemplars are useful for many strategies: multi-class, single class
# and KNN based. We already have the T5/KNN based solutions, we will look at Llama 8B/multiclass based solutions.
#


def skill_converter(retriever: ContextRetriever, skill_mode):
    if skill_mode == "desc":
        return DescExemplarConverter(retriever, InstanceMode.desc)
    if skill_mode == "exemplar":
        return DescExemplarConverter(retriever, InstanceMode.example)
    if skill_mode == "multi-class":
        return MultiClassSkillConverter(retriever)


# This creates factory
def build_skill_factory(output, factory, mode):
    context_retriever = load_context_retrievers(factory.schema, f"{output}/index/")
    skill_columns = [
        "id",
        "utterance",
        "template",
        "owner",
        "owner_mode",
        "arguments",
        "expectations",
    ]
    return ConvertedFactory(factory, [skill_converter(context_retriever, mode)], skill_columns)


def build_skill_dataset(output, factory, modes, index=True):
    # Save the things to disk first, for training we keep each module separate.
    # Down the road, we might
    build_desc_index(
        factory.tag,
        factory.schema,
        f"{output}/index/",
        EmbeddingStore.for_description(),
    )
    build_dataset_index(
        factory.tag,
        factory["train"],
        f"{output}/index/",
        EmbeddingStore.for_exemplar(),
    )

    for skill_mode in modes:
        prompted_factory = build_skill_factory(path, factory, mode=skill_mode, index=True)
        tags = ["train", "test", "validation"]
        for tag in tags:
            examples = prompted_factory[tag]
            with open(f"{path}/{promptManager.get_task_label(Task.SKILL)}.jsonl", "w") as file:
                print(f"there are {len(examples)} examples left for {tag}.")
                for example in examples:
                    file.write(f"{json.dumps(example)}\n")



if __name__ == "__main__":
    path = "./dugsets/sgd"
    tag = "sgd"

    factory = JsonDatasetFactory(path, tag)

    # this should build both desc and exemplar dataset
    skill_modes = ["desc", "exemplar"]
    #skill_modes = ["both"]
    build_skill_factory(path, factory, skill_modes)

