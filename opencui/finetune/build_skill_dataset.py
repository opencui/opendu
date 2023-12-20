import json
from opencui import EmbeddingStore, PromptedFactory, build_dataset_index, JsonDatasetFactory, \
    LugConfig, OneSkillTrainConverter, SkillTrainConverter, InstanceTrainConverter, InstanceMode
from opencui.core.retriever import build_desc_index, load_context_retrievers, ContextRetriever


def skill_converter(retriever: ContextRetriever):
    if LugConfig.skill_mode == "binary":
        return OneSkillTrainConverter(retriever)
    if LugConfig.skill_mode == "multiclass":
        return SkillTrainConverter(retriever)
    if LugConfig.skill_mode == "instance":
        return InstanceTrainConverter(retriever)
    if LugConfig.skill_mode == "instance.desc":
        return InstanceTrainConverter(retriever, InstanceMode.desc)
    if LugConfig.skill_mode == "instance.exemplar":
        return InstanceTrainConverter(retriever, InstanceMode.example)


def build_skill_factory(output, factory, index=True):
    # Save the things to disk first, for training we keep each module separate.
    # Down the road, we might
    if index:
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

    context_retriever = load_context_retrievers({factory.tag: factory.schema}, f"{output}/index/")
    return PromptedFactory(factory, [skill_converter(context_retriever)])


if __name__ == "__main__":
    path = "./datasets/sgd"
    tag = "sgd"
    factory = JsonDatasetFactory(path, tag)
    prompted_factory = build_skill_factory(path, factory, True)
    json.dump(prompted_factory.extra_tokens(), open(f"{path}/extra.tokens", "w"))
    tags = ["train", "test", "validation"]
    for tag in tags:
        examples = prompted_factory[tag]
        with open(f"{path}/{LugConfig.skill_mode}-{LugConfig.skill_prompt}.{tag}.jsonl", "w") as file:
            print(f"there are {len(examples)} examples left for {tag}.")
            for example in examples:
                file.write(f"{json.dumps(example)}\n")

