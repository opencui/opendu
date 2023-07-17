#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
import random
import sys
from collections import defaultdict
import gin


class SlotExpression:
    """
    expression examples
    """

    def __init__(self, service, intent, expression, slots, idx):
        self.utterance = expression
        self.intent = intent  # here it responds  to the certain active intent
        self.slots = slots  # dict to store slot, value pairs
        self.service = service
        self.idx = idx


class SlotExample:
    def __init__(self, source, utterance, slot_name, candidates, spans):
        self.source = source
        self.utterance = utterance
        self.slot_name = slot_name
        self.slot_description = None
        self.candidates = candidates  # this should include all the occurrences of the slot type.
        self.spans = spans

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def build_matcher(slot_values):
    matchers = dict()
    for name, values in slot_values.items():
        patterns = []
        for value in values:
            patterns.append(f"\\b{value}\\b")
        matchers[name] = re.compile('|'.join(patterns))
    return matchers


@gin.configurable
class GenerateSlotExamples:
    """
    generate examples
    """

    def __init__(self, input, output, training_percentage, neg_percentage, seed=None):
        if training_percentage < 0.0 or training_percentage > 1.0:
            raise ValueError("training_percentage is out of range")
        self.neg_percentage = neg_percentage
        self.training_percentage = training_percentage
        self.seed = seed
        self.all_sample = []
        self.used_template = set()
        self.base_path = input
        self.output = output

    def load_intent_to_slots(self):
        slot_index = defaultdict(list)
        with open(self.base_path + 'schema.json', encoding='utf-8') as f:
            f = json.load(f)
            for service in f:
                # in the "overall"   generate mode,only pick <service>_1  if there are multiple  services for one intent
                if service["service_name"][-1] != '1':
                    continue

                for intent in service['intents']:
                    slot_index[intent['name']] = []
                    for name in intent['required_slots']:
                        for slot in service['slots']:
                            if slot['name'] == name:
                                # if slot['type']  !=  'System.Boolean':#delete the slot which value type is  boolean
                                slot_index[intent['name']].append(name)
                    for name in intent['optional_slots'].keys():
                        for slot in service['slots']:
                            if slot['name'] == name:
                                # if slot['type']  !=  'System.Boolean':
                                slot_index[intent['name']].append(name)
                    # This is used to dedup the slot, now we assume the slot are global.
                    slot_index[intent['name']] = list(set(slot_index[intent['name']]))
        return slot_index

    def load_slot_values(self):
        """
        Load all the values for every slot in this dataset.
        """
        files = os.listdir(self.base_path)
        values = defaultdict(set)
        for file in files:
            if file[:6] != 'dialog':
                continue
            # if file   =="dialogues_074.json":
            with open(self.base_path + file, encoding='utf-8') as f:
                f = json.load(f)
                for dialogue in f:
                    # only use the additional slots in ['slot_values']
                    pre_slot_name = set()
                    pre_slot = dict()
                    for turn in dialogue['turns']:
                        # do not use the utterance that has more than 1 frame
                        if turn['speaker'] != 'USER':
                            continue

                        if len(turn['frames']) > 1:
                            continue

                        frame = turn['frames'][0]
                        if frame['service'][-1] != '1':
                            continue

                        # all the text are lowercase
                        for slot_name, slot_val_list in frame['state']['slot_values'].items():
                            for idx in range(len(slot_val_list)):
                                values[slot_name].add(slot_val_list[idx].lower())
        return values

    def load_slot_expressions(self):
        """
        load original sgd data and create expression examples
        :param base_path: input path to original sgd dataset
        :return: expression examples
        """
        files = os.listdir(self.base_path)
        expressions = list()
        for file in files:
            if file[:6] != 'dialog':
                continue
            # if file   =="dialogues_074.json":
            with open(self.base_path + file, encoding='utf-8') as f:
                f = json.load(f)
                for dialogue in f:
                    # only use the additional slots in ['slot_values']
                    pre_slot_name = set()
                    pre_slot = dict()
                    for turn in dialogue['turns']:
                        # do not use the utterance that has more than 1 frame
                        if turn['speaker'] != 'USER':
                            continue

                        if len(turn['frames']) > 1:
                            continue

                        all_frame_slot_name = set()
                        all_slot = dict()

                        frame = turn['frames'][0]
                        if frame['service'][-1] != '1':
                            continue

                        # all the text are lowercase
                        service = frame['service']
                        intent = frame['state']['active_intent']
                        utterance = turn['utterance'].lower()

                        slots = defaultdict(list)

                        for _slot in frame['slots']:
                            slots[_slot['slot']].append((_slot['start'], _slot['exclusive_end']))

                        expressions.append(SlotExpression(service, intent, utterance, slots, dialogue['dialogue_id']))
        return expressions

    def build_examples(self, expressions, intent_slots, matchers):
        examples = defaultdict(list)
        random.seed(self.seed)
        total_pos_cnt = 0
        total_neg_cnt = 0

        for expression in expressions:
            # print("generate expression")
            intent = expression.intent
            utterance = expression.utterance

            all_slot_names = intent_slots[intent]
            for slot_name in all_slot_names:
                # This means that we have value for this slot in this utterance.
                matcher = matchers[slot_name]
                candidate_spans = []
                matches = matcher.finditer(utterance)
                for match in matches:
                    candidate_spans.append(match.span())

                if slot_name in expression.slots.keys():
                    value_spans = expression.slots[slot_name]
                    total_pos_cnt += 1
                    examples[intent].append(SlotExample(intent, utterance, slot_name, candidate_spans, value_spans))
                else:
                    value_spans = []
                    total_neg_cnt += 1
                    examples[intent].append(SlotExample(intent, utterance, slot_name, candidate_spans, value_spans))
        print(f'total_pos_cnt:{total_pos_cnt}  total_neg_cnt:{total_neg_cnt}  total:{total_pos_cnt + total_neg_cnt}')
        return examples

    def __call__(self):
        intent_slots = self.load_intent_to_slots()
        slot_values = self.load_slot_values()
        matchers = build_matcher(slot_values)
        expressions = self.load_slot_expressions()

        examples = self.build_examples(expressions, intent_slots, matchers)
        self.save(examples)

    def save(self, slot_examples):
        """
        save generated examples into tsv files
        :param slot_examples:  generated examples
        :param path: output path
        :return: None
        """
        for key, examples in slot_examples.items():
            # key is train or test
            with open(f"{self.output}/new_slot.json", 'w') as f:
                for example in examples:
                    f.write(json.dumps(example.toJSON(), indent=4) + "\n")
        return


if __name__ == '__main__':
    gin.parse_config_file(sys.argv[1])

    build = GenerateSlotExamples()
    build()
