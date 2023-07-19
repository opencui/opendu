import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from string import Template


class SlotMeta:
    """
    For now, we assume the
    """
    def __init__(self, name, description, prompt=None):
        self.name = name
        self.description = description
        self.prompt = prompt


class Reference:
    def __init__(self, kind, reference):
        """
        With kind to be description or exemplar for Englished. This can change.
        """
        self.kind = kind
        self.reference = reference


class Decoder:
    """
    This is the low level decoder that higher level nlu rely on.
    """
    def __init__(self, peft_model_id, device="cuda"):
        self.config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name_or_path)

        self.model = PeftModel.from_pretrained(model, peft_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path, add_eos_token=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.model.to(device)
        self.slot_name_templates = {
            "en": Template("What is the value for $slot in '$utterance'?")
        }
        self.slot_prompt_templates = {
            "en": Template("The question is $prompt. Does $utterance means yes or no?")
        }
        self.referent_templates = {

        }

    def generate(self, inputs):
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                eos_token_id=3)
        return self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

    def fill_entity_slots(self, utterance, p_slot_metas, lang="en"):
        """
        Given the utterance and a list of slot metas, we encode them into what we need before call generate.
        """
        inputs = []
        for meta in p_slot_metas:
            if meta.prompt:
                inputs.append(self.slot_prompt_templates[lang].substitute({"prompt": meta.prompt, "utterance": utterance}))
            else:
                inputs.append(self.slot_name_templates[lang].substitute({"slot": meta.name, "utterance": utterance}))
        return self.generate(inputs)

    def extract_function(self, utterance, references, lang="en"):
        """
        Given a list of references, we return a list of label.
        """
        inputs = []
        for reference in references:
            inputs.append(self.referent_templates[lang].substitute({
                "utterance": utterance,
                "kind": reference.kind,
                "reference": reference.reference
            }))
        return self.generate(inputs)


if __name__ == "__main__":
    decoder = Decoder("OpenCUI/test_PROMPT_TUNING_CAUSAL_LM")
    slot_metas = [
        SlotMeta("event_date", ""),
        SlotMeta("destination", "")
    ]
    utterance = "tell me what events i have scheduled in my calendar for the 13th of this month."
    print(decoder.fill_entity_slots(utterance, slot_metas))