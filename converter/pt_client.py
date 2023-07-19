import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class SlotMeta:
    """
    For now, we assume the
    """
    def __init__(self, name, description, slot_type, prompt):
        self.name = name
        self.description = description
        self.slot_type = slot_type
        self.prompt = prompt


class FunctionId:
    """
    If the function is templated, then function should be template, so that we can specialize it.
    For example: f"slot_update{slot_type}"
    """
    def __init__(self, service, function, type_parameters=[]):
        self.service = service
        self.function = function
        self.type_parameters = type_parameters


class Reference:
    def __init__(self, kind, reference):
        """
        With kind to be description or exemplar for Englished. This can change.
        """
        self.kind = kind
        self.reference = reference

class Services:
    def list_functions(self, utterance):
        """
        Return the list of function.
        """
        return None

    def list_slot_metas(self, function):
        return None


class Decoder:
    """
    This is the low level decoder that higher level nlu rely on.
    """
    def __init__(self, peft_model_id, device):
        self.config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name_or_path)

        self.model = PeftModel.from_pretrained(model, peft_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path, add_eos_token=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.model.to(device)

    def generate(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                eos_token_id=3)
        return self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

    def fill_entity_slots(self, utterance, slot_metas):
        """
        Given the utterance and a list of slot metas, we encode them into what we need before call generate.
        """
        inputs = []
        return self.generate(inputs)

    def extract_function(self, utterance, references):
        """
        Given a list of references, we return a list of label.
        """
        inputs = []
        return self.generate(inputs)