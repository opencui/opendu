import json
from datasets import Dataset, load_dataset
from opencui.finetune.commons import PromptedFactory
from opencui.core.config import LugConfig
from opencui.core.prompt import ExtractiveSlotPrompts

class ConllLabel:
    label_info = {
            "PER" : {"name": "person"},
            "LOC" : {"name": "location"},
            "ORG" : {"name": "organization"}
        }
    # One need to make sure that label encoded in this order.
    id_to_label = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    def __init__(self, label):
        self.labels = ConllLabel.id_to_label[int(label)].split("-")

    def is_payload(self):
        return len(self.labels) != 1

    def is_start(self):
        return self.is_payload() and self.labels[0] == "B"

    def payload(self):
        return self.labels[-1]

    def is_close(self, last):
        if self.is_start():
            return True
        return self.payload() != last.payload()

    def get_name(self):
        return ConllLabel.label_info[self.payload()]["name"]


class ConllLabelBuilder:
    def __init__(self, cares):
        self.sep = "|"
        self.start = "["
        self.end = ']'
        self.cares = cares

    def care(self, label: ConllLabel):
        return label.payload() in self.cares

    def __call__(self, tokens, tags):
        print("fuck")
        out = []
        last_label = None
        for index, tag in enumerate(tags):
            label = ConllLabel(tag)
            # We need to make two decisions, whether to add start marker, whether to add end marker.
            if last_label is not None and label.is_close(last_label) and self.care(last_label):
                out.append(self.sep)
                out.append(last_label.get_name())
                out.append(self.end)

            if label.is_start() and self.care(label):
                out.append(self.start)

            out.append(tokens[index])
            last_label = label

        if self.care(last_label):
            out.append(self.sep)
            out.append(last_label.get_name())
            out.append(self.end)

        return " ".join(out)

    def good(self, tags):
        for tag in tags:
            label = ConllLabel(tag)
            if self.care(label): return True
        return False


class Conll03OneSlotConverter(TrainConverter, ABC):
    def __init__(self, prompt, care):
        self.prompt = prompt
        self.care = care
        self.build_label = ConllLabelBuilder([self.care])

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, tokens in enumerate(batch["tokens"]):
            tags = batch["ner_tags"][idx]
            input_dict = {"utterance": " ".join(tokens)}
            input_dict.update(ConllLabel.label_info[self.care])

            # without the values for conll.
            input_dict["values"] = []
            if self.build_label.good(tags):
                ins.append(self.prompt(input_dict))
                outs.append(f"{self.build_label(tokens, tags)}</s>")



if __name__ == "__main__":
    path = "./datasets/conllner"
    factory = load_dataset('conll2003')
    converter = Conll03OneSlotConverter(ExtractiveSlotPrompts[LugConfig.get().slot_prompt], "PER")
    prompted_factory = PromptedFactory(factory, [converter], ["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"])

    tags = ["train", "test", "validation"]
    for tag in tags:
        examples = prompted_factory[tag]
        with open(f"{path}/{tag}.jsonl", "w") as file:
            print(f"there are {len(examples)} examples left for {tag}.")
            for example in examples:
                file.write(f"{json.dumps(example)}\n")
