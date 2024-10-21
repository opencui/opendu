import json
from opendu.core.embedding import EmbeddingStore
from opendu import ConvertedFactory, build_dataset_index, JsonDatasetFactory, skill_converter
from opendu.core.retriever import build_desc_index, load_context_retrievers
from opendu.core.prompt import promptManager1, Task


#
# Before we build dataset for skills, we need to handle the following steps:
# 1. extract schema.
# 2. extract exemplars: both for skill and slots.
# 3. extract entities.
# 4. build index.
# The separate indexing for desc and exemplars are useful for many strategies: multi-class, single class
# and KNN based. We already have the T5/KNN based solutions, we will look at Llama 8B/multiclass based solutions.
#

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


# This is how we create the skill dataset given exemplars dataset.
def build_skill_dataset(output, factory, modes, index=True):
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

    print("Now we create dataset.")
    for skill_mode in modes:
        prompted_factory = build_skill_factory(output, factory, mode=skill_mode)
        tags = ["train"] #, "test", "validation"]
        for tag in tags:
            examples = prompted_factory[tag]
            with open(f"{output}/{promptManager1.get_task_label(Task.SKILL)}.jsonl", "w") as file:
                print(f"there are {len(examples)} examples left for {tag}.")
                for example in examples:
                    file.write(f"{json.dumps(example)}\n")



if __name__ == "__main__":
    # This showes how we create skill dataset from
    ds_path = "./dugsets/sgd"
    tag = "sgd"

    factory = JsonDatasetFactory(ds_path, tag)

    # this should build both desc and exemplar dataset
    skill_modes = ["rag"]
    #skill_modes = ["both"]
    build_skill_dataset(ds_path, factory, skill_modes, index=False)

