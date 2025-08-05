import json

from opendu.finetune.structure_converter import YniConverter
from opendu.core.config import RauConfig
from opendu.finetune.commons import (MergedDatasetFactory, FtDatasetFactory, DatasetFactory)


def build_converter(mode, path):
    converter = None
    columns = None
    if mode == "yni":
        converter = YniConverter()
        columns = ["context", "question", "response", "label"]
    return converter, columns


def load_merged_factories(paths, mode)-> DatasetFactory:
    factories = []
    prompt = RauConfig.get().prompt[mode]
    for path in paths:
        print(f"processing {path}")
        converter, columns = build_converter(mode, path)
        ds = FtDatasetFactory(path, [converter], columns)
        factories.append(ds)
    return MergedDatasetFactory(factories)


def print_ds(ds):
    count = 0
    for item in ds:
        print(json.dumps(item, indent=2))
        count += 1
    print(count)


# Load file tune set, based on what is inside the --training_mode desc-exemplar-extractive-slot
def load_training_dataset(training_mode, debug=False):
    factory = None
    # load_bot_factory(converted_factories)
    print(training_mode)
    if training_mode == "yni":
        datasets = ["ftds/yni/circa/", "ftds/yni/ludwig", "ftds/yni/chang"]
        print(f"load yni datasets: {datasets}")
        factory = load_merged_factories(datasets, training_mode)
    if training_mode == "sf_ss":
        print(f"load sf_ss datasets")

    assert factory != None

    # If we debug dataset, we do not train.
    if debug:
        print_ds(factory["train"])
        print_ds(factory["test"])
        print_ds(factory["dev"])
        exit(0)
    return factory


if __name__ == "__main__":
        # Turn off the evaluation mode, but why?
    RauConfig.get().eval_mode = False
    load_training_dataset("yni", True)