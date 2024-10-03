# We need to introduce a list of special tokens so that they can impact the response better.
from enum import Enum


class SpecialTokens(str, Enum):
    by_description = "<description>"
    by_exemplar = "<exemplar>"
    begin_template = "<template>"
    end_template = "</template>"
    begin_utterance = "<utterance>"
    end_utterance = "</utterance>"
    begin_func = "<func>"
    end_func = "</func>"
    begin_func_name = "<func_name>"
    end_func_name = "</func_name>"
    begin_func_desc = "<func_desc>"
    end_func_desc = "</func_desc>"
    begin_exemplar = "<exemplar>"
    end_exemplar = "</exemplar>"
    begin_context = "<context>"
    end_context = "</context>"
    begin_tf_response = "<true_false_response>"
    end_tf_response = "</true_face_response>"
    begin_yn_response = "<yes_no_response>"
    end_yn_response = "</yes_no_response>"
    begin_slot = "<slot>"
    end_slot = "</slot>"
    begin_slot_name = "<slot_name>"
    end_slot_name = "</slot_name>"
    begin_slot_desc = "<slot_desc>"
    end_slot_desc = "</slot_desc>"
    @classmethod
    def list(cls):
        return [c.value for c in cls]