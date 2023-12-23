import json
from opencui import EmbeddingStore, PromptedFactory, build_dataset_index, JsonDatasetFactory, \
    LugConfig, InstanceTrainConverter, InstanceMode
from opencui.core.retriever import build_desc_index, load_context_retrievers, ContextRetriever


def skill_converter(retriever: ContextRetriever, skill_mode):
    if skill_mode == "desc":
        return InstanceTrainConverter(retriever, InstanceMode.desc)
    if skill_mode == "exemplar":
        return InstanceTrainConverter(retriever, InstanceMode.example)


def build_skill_factory(output, factory, mode, index=True):
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
    return PromptedFactory(factory, [skill_converter(context_retriever, mode)])


if __name__ == "__main__":
    path = "./datasets/sgd"
    tag = "sgd"

    factory = JsonDatasetFactory(path, tag)
    # this should build both desc and exemplar dataset
    for skill_mode in ["desc", "exemplar"]:
        prompted_factory = build_skill_factory(path, factory, mode=skill_mode, index=True)
        tags = ["train", "test", "validation"]
        for tag in tags:
            examples = prompted_factory[tag]
            with open(f"{path}/{skill_mode}-{LugConfig.skill_prompt}.{tag}.jsonl", "w") as file:
                print(f"there are {len(examples)} examples left for {tag}.")
                for example in examples:
                    file.write(f"{json.dumps(example)}\n")

