import json

from datasets import load_dataset

from opencui import (JsonDatasetFactory, LugConfig, OneSlotExtractConverter,
                     collect_slot_values, ExtractiveSlotPrompts,
                     PromptedFactory, NliConverter, NliPrompts, MappedDatasetDict)


# Here we create the dataset factory for skills
def build_skill_factory(skill_modes, factories):
    # make sure run build_skill_dataset first.
    for skill_mode in skill_modes:
        factories.append(
            JsonDatasetFactory("./datasets/sgd/", "sgd", f"{skill_mode}-{LugConfig.skill_prompt}.")
        )


def build_extractive_slot_factory(converted_factories):
    factories = [
        JsonDatasetFactory("./datasets/sgd/", "sgd"),
    ]
    for index, factory in enumerate(factories):
        entity_values = collect_slot_values(factory.__getitem__("train"))
        slot_converter = OneSlotExtractConverter(
            factory.schema, ExtractiveSlotPrompts[LugConfig.slot_prompt], entity_values
        )
        converted_factories.append(PromptedFactory(factory, [slot_converter]))


def build_nli_factory(converted_factories):
    # Here we assume the raw input is sentence, focus and label (positive, negative and neutral)
    semeval2016 = load_dataset("glue", "mnli")
    factories = [MappedDatasetDict(semeval2016, "validation_matched", "validation_mismatched")]
    for index, factory in enumerate(factories):
        converter = NliConverter(NliPrompts[LugConfig.nli_prompt])
        converted_factories.append(PromptedFactory(factory, [converter], []))


# Load training set, based on what is inside the --training_mode desc-exemplar-extractive-slot
def load_training_dataset(args):
    converted_factories = []
    if "desc" in args.training_mode:
        print("load desc dataset")
        build_skill_factory(["desc"], converted_factories)
    if "exemplar" in args.training_mode:
        print("load exemplar dataset")
        build_skill_factory(["exemplar"], converted_factories)
    if "extractive_slot" in args.training_mode:
        print("load slot dataset")
        build_extractive_slot_factory(converted_factories)
    if "nli" in args.training_mode:
        print("load nli dataset")
        build_nli_factory(converted_factories)

    assert len(converted_factories) != 0

    # If we debug dataset, we do not train.
    if args.debug_dataset:
        count = 0
        for factory in converted_factories:
            ds = factory["train"]
            for item in ds:
                print(json.dumps(item, indent=2))
                count += 1
        print(count)
        exit(0)
    return converted_factories


def print_factories(factories):
    for factory in factories:
        ds = factory.__getitem__("train")
        count = 0
        for item in ds:
            print(item)
            count += 1
        print(f"There are {count} instances")
