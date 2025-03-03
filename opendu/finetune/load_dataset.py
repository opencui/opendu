import json
import glob


from opendu.core.config import RauConfig
from opendu.finetune.structure_converter import YniConverter
from opendu.finetune.commons import (JsonBareDatasetFactory, RawJsonDatasetFactory, JsonDatasetFactory,
                                     DatasetFactoryMerger, ConvertedFactory, PromptedFactory)


# Here we create the dataset factory for skills
def load_skill_factory(skill_modes, factories, suffix=""):
    # make sure run build_skill_dataset first.
    for skill_mode in skill_modes:
        factories.append(
            JsonDatasetFactory("./dugsets/sgd/", "sgd", f"{RauConfig.get().skill_prompt}{suffix}")
        )

# Here we create the dataset factory for skills
def load_id_bc_factory(mode, factories, suffix=""):
    # make sure run build_skill_dataset first.
    factories.append(
        RawJsonDatasetFactory("./dugsets/sgd/", "sgd", f"{mode}")
    )


def load_extractive_slot_factory(converted_factories):
    converted_factories.append(
        DatasetFactoryMerger([
            JsonBareDatasetFactory("./dugsets/sgd/", "sgd", "slots-"),
            JsonBareDatasetFactory("./dugsets/conll03/", "ner"),
        ])
    )


def load_yni_factory(converted_factories):
    # Here we assume the raw input is sentence, focus and label (positive, negative and neutral)
    converter = YniConverter()
    columns = ["context", "question", "response", "label"]
    converted_factories.append(
        ConvertedFactory(JsonBareDatasetFactory("./dugsets/yni/", "yni"), [converter], columns)
    )


def load_bot_factory(converted_factories):
    # this is used to extract all the datasets from labeling process and make it available.
    # We assume that there are botsets/{lang}/{bots}/
    matching_data_directories = glob.glob("./botsets/en/*/MatchLabeledData.json")
    columns = ['_created_at', '_id', 'bot', 'context', 'decision', 'lang', 'matchType', 'owner', 'reference', 'userId', 'userOrg', 'utterance']
    for directory in matching_data_directories:
        # Add prompt to it.
        converted_factories.append(PromptedFactory(directory, columns))


# Load training set, based on what is inside the --training_mode desc-exemplar-extractive-slot
def load_training_dataset(args):
    converted_factories = []
    # load_bot_factory(converted_factories)
    training_modes = set(args.training_mode.split(","))
    for training_mode in training_modes:
        print(training_mode)
        if "id_mc" in training_mode:
            load_id_bc_factory(training_mode, converted_factories)
        if "desc" in training_mode:
            print("load desc dataset")
            load_skill_factory(["desc"], converted_factories)
        if "exemplar" in training_mode:
            print("load exemplar dataset")
            load_skill_factory(["exemplar"], converted_factories)
        if "extractive_slot" in training_mode:
            print("load slot dataset")
            load_extractive_slot_factory(converted_factories)
        if "nli" in training_mode:
            print("load nli dataset")
            load_yni_factory(converted_factories)

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
