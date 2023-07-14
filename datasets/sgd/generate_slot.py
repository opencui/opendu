#!/usr/bin/env python
# coding: utf-8


import json
import os
import random
from collections import defaultdict
from absl import app
from absl import flags

flags.DEFINE_bool(
    'fix_random_seed', True,
    'use fixed random seed(for debug)')

flags.DEFINE_string(
    'input', './train/',
    'path to input file (original dataset)'
    )

flags.DEFINE_string(
    'output', './res/train/',
    'output path to save generated intent model examples'
    )

flags.DEFINE_float(
    'negative_proportions', 1.0,
    'how many negative examples to generate for each positive example')

flags.DEFINE_float(
    'training_percentage', 1.0,
    'how many examples for training')

flags.DEFINE_float(
    'dev_percentage', 0.0,
    'how many examples for dev')


FLAGS = flags.FLAGS

MODEL = "slots"
TRAIN = "train"
TEST = "test"
DEV = "dev"

_1toone={"0":"zero ", "1":"one", "2":"two", "3":"three", "4":"four", "5":"five", "6":"six", "7":"seven",
                "8":"eight", "9":"nine", "10":"ten", "11":"eleven", "12":"twelve", "13":"thirteen",
                 }
_oneto1={
'zero': "0",
'one': "1",
'two': "2",
'three': "3",
'four': "4",
'five': "5",
'six': "6",
'seven': "7",
'eight': "8",
'nine': "9",
'ten': "10",
'eleven': "11",
'twelve': "12",
'thirteen': "13",
}
arbic_set=set(_1toone.keys())
word_set=set(_1toone.values())
NUMBER_SET=set( _1toone.keys()) | set(_1toone.values())


def dataset_type(train_percentage, dev_percentage):
    val = random.random()
    if val < train_percentage:
        return TRAIN
    elif val < (train_percentage + dev_percentage):
        return DEV
    return TEST

class Expression:
    """
    expression examples
    """
    def __init__(self, expression, intent, slots,service,string_list,vague_slot_names,id):
        self.utterance = expression
        self.intent = intent # here it responds  to the certain active intent
        self.slots = slots # dict to store slot, value pairs
        self.service=service
        self.string_list=string_list
        self.template=""
        self.vague_slot_names=vague_slot_names
        self.id=id


class IntentTemplate:
    """
    restore the all template of a certain intents, including the set of all possible examplers ,and the dict for all slot 
    """
    def __init__(self,service):
        self.exampler_set=set()
        self.slot_dict=defaultdict(set)
        self.service=service
        self.val_len_range=dict()

    def generate_expression_template(self,slot_dict, utterance,string_list):
        '''
        replacing the slot val with the slot name,to avoid match the short slot val which may be inclued in other long slot val,we need sort by the length of the slot val
        '''
        if string_list == []:
            return utterance
        single_dict = dict()

        for key, values in slot_dict.items():
            for value in values:
                single_dict[value] = key

        string_list=sorted(string_list,key=lambda x:x[0])
        res_utterance=utterance[:string_list[0][0]]
        for i,(cur_start,cur_end) in enumerate(string_list) :

            res_utterance = res_utterance+'|||'+single_dict[utterance[cur_start:cur_end]]+'|||'
            if i == len(string_list)-1 :
                res_utterance=res_utterance+utterance[cur_end:]
            else :
                res_utterance = res_utterance+utterance[cur_end:string_list[i+1][0]]
        return res_utterance



    def add_sample(self,expression):
        expression_template=self.generate_expression_template(expression.slots,expression.utterance,expression.string_list)
        expression.template=expression_template
        self.exampler_set.add(expression_template)
        for slot_name,slot_val_list  in expression.slots.items():
            for slot_val in slot_val_list:
                self.slot_dict[slot_name].add(slot_val)

    def slot_val_statics(self):
        for slot_name,slot_vals in self.slot_dict.items():
            min_len=100
            max_len=0
            for slot_val in slot_vals:
                min_len=min(len(slot_val.split()),min_len)
                max_len=max(len(slot_val.split()),max_len)
            self.val_len_range[slot_name]=(min_len,max_len)

    def generate_equal_sample(self,expression):
        expression_template=expression.template

        for slot_name,slot_vals  in self.slot_dict.items():
            if '|||'+slot_name+'|||'   in  expression_template:

                expression_template=expression_template.replace('|||'+slot_name+'|||',list(slot_vals)[random.randint(0,len(slot_vals)-1)])

        return expression_template


    def generate_equal_sample_fixed(self,expression,fixed_slot_name,fixed_slot_val,masked_len=0):
        '''
        this fun could generate a equal utterance when giving a certain exampler and in which has a fixed slot value that could not be changed
        '''
        expression_template=expression.template
        for slot_name,slot_vals  in self.slot_dict.items():
            if '|||'+slot_name+'|||'   in  expression_template   and  expression_template.find('|||'+slot_name+'|||') != expression_template.find('|||'+fixed_slot_name+'|||'):
                expression_template=expression_template.replace('|||'+slot_name+'|||',list(slot_vals)[random.randint(0,len(slot_vals)-1)])
        new_start=expression_template.find('|||'+fixed_slot_name+'|||')
        if masked_len ==0:
            new_end=new_start+len(fixed_slot_val)
            expression_template=expression_template.replace('|||'+fixed_slot_name+'|||',fixed_slot_val)
        else:
            mask_str=""
            for i in range(masked_len):
                if i == 0:
                    mask_str+="[MASK]"
                else:
                    mask_str+=" [MASK]"
            new_end =len(mask_str) + new_start
            expression_template=expression_template.replace('|||'+fixed_slot_name+'|||',mask_str)
        return expression_template,(new_start,new_end)

def generate_slotindex(base_path):
    slot_index = defaultdict(list)
    with open(base_path + 'schema.json', encoding='utf-8') as f:
        f = json.load(f)
        for service in f:
            # in the "overall"   generate mode,only pick <service>_1  if there are multiple  services for one intent
            if service["service_name"][-1]!= '1':
                continue

            for intent in service['intents']:
                slot_index[intent['name']] = []
                for name in intent['required_slots']:
                    for slot in service['slots']:
                        if slot['name'] == name:
                            #if slot['type']  !=  'System.Boolean':#delete the slot which value type is  boolean
                            slot_index[intent['name']].append(name)
                for name in intent['optional_slots'].keys():
                    for slot in service['slots']:
                        if slot['name'] == name:
                            #if slot['type']  !=  'System.Boolean':
                            slot_index[intent['name']].append(name)
                slot_index[intent['name']] = list(set(slot_index[intent['name']]))

    return slot_index



def judge_legal_loc(utterance,pattern):
    '''
    to judge if the pattern have a legal loc in the utterance 
    '''
    starts=set()
    for i in range(len(utterance)):
        if utterance.find(pattern,i) != -1:
            starts.add(utterance.find(pattern,i))
    starts=list(starts)

    ends = [start + len(pattern) - 1 for start in starts]
    span = [(start, end) for start, end in zip(starts, ends)]
    for idx, (start, end) in enumerate(span):

        if (start == 0 or utterance[start - 1] == ' ' or  utterance[start - 1] in '!"#$%&\'()*+,-./;<=>?@[\\]^_`{|}~') and (
                end == len(utterance) - 1 or utterance[end + 1] == ' ' or utterance[end + 1] in '!"#$%&\'()*+,-./;<=>?@[\\]^_`{|}~'):

                return True

    return False


def judge_single_word(utterance, pattern):
    '''
    give the utterance and the pattern,return all the legal location of the pattern in utterance
    '''
    if  pattern in NUMBER_SET:
        if pattern in arbic_set :
            if not  judge_legal_loc(utterance,pattern)  and judge_legal_loc(utterance,_1toone[pattern]):
                print(f"pattern ||{pattern}||  not  in ||{utterance}||,  but ||{_1toone[pattern]}|| in it.")
                pattern=_1toone[pattern]
        elif pattern in word_set:
            if not  judge_legal_loc(utterance,pattern)  and judge_legal_loc(utterance,_oneto1[pattern]):
                print(f"pattern ||{pattern}||  not  in ||{utterance}||,  but ||{_oneto1[pattern]}|| in it.")
                pattern=_oneto1[pattern]
    if not utterance.count(pattern):
        return []

    starts=set()
    for i in range(len(utterance)):
        if utterance.find(pattern,i) != -1:
            starts.add(utterance.find(pattern,i))
    starts=list(starts)

    ends = [start + len(pattern) - 1 for start in starts]
    span = [(start, end) for start, end in zip(starts, ends)]
    label = [0] * len(starts)
    res = []
    for idx, (start, end) in enumerate(span):
        if (start == 0 or utterance[start - 1] == ' ' or  utterance[start - 1] in '!"#$%&\'()*+,-./;<=>?@[\\]^_`{|}~') and (
                end == len(utterance) - 1 or utterance[end + 1] == ' ' or utterance[end + 1] in '!"#$%&\'()*+,-./;<=>?@[\\]^_`{|}~'):
            label[idx] = 1
    for i in range(len(label)):
        if label[i] == 1:
            res.append(span[i])
    return res

def load_add_slot_value(base_path,SLOTS):
    """
    load original sgd data and create expression examples
    :param path: input path to original sgd dataset
    :return: expression examples
    """

    files = os.listdir(base_path)
    expressions = list()
    intent_template_dict=dict()
    for file in files:
        if file[:6] == 'dialog':
        # if file   =="dialogues_074.json":
            with open(base_path + file, encoding='utf-8') as f:
                f = json.load(f)
                for dialogue in f:
                    #only use the additional slots in ['slot_values']
                    pre_slot_name=set()
                    pre_slot=dict()
                    for  turn in  dialogue['turns']:
                        #do not use the utterance that has more than 1 frame
                        if len(turn['frames'])  >1 :
                            continue
                        if turn['speaker'] == 'USER' :
                            all_frame_slot_name=set()
                            all_slot=dict()

                            for i,frame in enumerate(turn['frames']):

                                if frame['service'][-1] == '1':

                                    # all the text are lowercase
                                    turn['utterance']=turn['utterance'].lower()
                                    for slot_name,slot_val_list  in  frame['state']['slot_values'].items():
                                        for i in range(len(slot_val_list)):
                                            slot_val_list[i]=slot_val_list[i].lower()


                                    cur_slot_name=set()
                                    string_list=[]
                                    slots = defaultdict(dict)

                                    for _slot in frame['slots']:
                                        slots[_slot['slot']][turn['utterance'][_slot['start']:_slot['exclusive_end']]]=(_slot['start'],_slot['exclusive_end'])
                                        string_list.append((_slot['start'],_slot['exclusive_end']))

                                    old_slots=slots.copy()

                                    vague_slot_names=set()
                                    new_slot_val2slot_name=defaultdict(list)
                                    for slot_name,slot_val_list  in  frame['state']['slot_values'].items():
                                        #make the static for the current slot names
                                        cur_slot_name.add(slot_name)

                                    for slot_name,slot_val_list  in  frame['state']['slot_values'].items():
                                        for i in range(len(slot_val_list)):
                                            utterance=turn['utterance']
                                            pattern=slot_val_list[i]
                                            #check the data in the slot_values whether has the condition like 1 <->  one
                                            if  pattern in NUMBER_SET:
                                                if pattern in arbic_set :
                                                    if not  judge_legal_loc(utterance,pattern)  and judge_legal_loc(utterance,_1toone[pattern]):
                                                        pattern=_1toone[pattern]
                                                elif pattern in word_set:
                                                    if not  judge_legal_loc(utterance,pattern)  and judge_legal_loc(utterance,_oneto1[pattern]):
                                                        pattern=_oneto1[pattern]
                                            slot_val_list[i]=pattern
                                    #the slot that would be used in the "slot_values"  is the new slot name ,or the slot values has been changed
                                    for slot_name,slot_val_list  in  frame['state']['slot_values'].items():
                                        if slot_name in cur_slot_name-pre_slot_name:
                                            for slot_val in slot_val_list:
                                                new_slot_val2slot_name[slot_val].append(slot_name)

                                    for slot_name,slot_val_list  in  frame['state']['slot_values'].items():
                                        if slot_name in cur_slot_name&pre_slot_name:
                                            for slot_val in slot_val_list:
                                                if slot_val in set(frame['state']['slot_values'][slot_name])-set(pre_slot[slot_name]):
                                                    #if the slot_val is changed ,we would also use it
                                                    new_slot_val2slot_name[slot_val].append(slot_name)

                                    for slot_name,slot_val_list  in  frame['state']['slot_values'].items():
                                        if slot_name in cur_slot_name-pre_slot_name:
                                            for slot_val in slot_val_list:
                                                #if the slot val in the utterance locates more than 1 ,then  we could not judge which corresponds to it ,so we add it into the vague list
                                                if len(judge_single_word(turn['utterance'],slot_val))  >1 :
                                                    vague_slot_names.add(slot_name)

                                    for slot_val,slot_name_list in new_slot_val2slot_name.items():
                                        #if in the new slot list exists one value corresponds to more than one slot name, we could not judge the which it belongs to ,so we  need to ignore it
                                        if len(slot_name_list) >1  :
                                            for item in slot_name_list:
                                                vague_slot_names.add(item)

                                    single_dict = dict()
                                    for slot_name,slot_val_list in frame['state']['slot_values'].items():
                                        if slot_name in cur_slot_name-pre_slot_name and  (slot_name not in vague_slot_names):
                                            for value in slot_val_list:
                                                single_dict[value] = slot_name

                                    for slot_name,slot_val_list in frame['state']['slot_values'].items():
                                        if slot_name in cur_slot_name&pre_slot_name and  (slot_name not in vague_slot_names):
                                            for value in slot_val_list:
                                                if value in set(frame['state']['slot_values'][slot_name])-set(pre_slot[slot_name]):
                                                    single_dict[value] = slot_name

                                    #replace the long slot value first to avoid the error
                                    single_dict = sorted(single_dict.items(), key=lambda x: len(x[0]), reverse=True)


                                    for slot_val,slot_name  in single_dict:
                                        if  len(judge_single_word(turn['utterance'],slot_val)) ==1  and  (slot_name not in slots)  and  \
                                            ((slot_name in cur_slot_name-pre_slot_name  and slot_name  not in vague_slot_names)  or  \
                                            (slot_name in cur_slot_name&pre_slot_name and slot_val in set(frame['state']['slot_values'][slot_name])-set(pre_slot[slot_name])))  :
                                                becovered=False
                                                start_position=judge_single_word(turn['utterance'],slot_val)[0][0]
                                                end_position=start_position+len(slot_val)
                                                for _,pre_slot_val_list in slots.items():
                                                        for pre_slot_str,pre_span in pre_slot_val_list.items():
                                                            if  slot_val in pre_slot_str and  start_position >=pre_span[0] and end_position <=pre_span[1]:
                                                                becovered=True
                                                if becovered==False:
                                                    slots[slot_name][slot_val]=(start_position,end_position)
                                                    string_list.append((start_position,end_position))

                                    vague_slot_names=vague_slot_names-set(old_slots.keys()) #the slot val that been flaged in the slots should be used


                                    all_frame_slot_name=all_frame_slot_name|cur_slot_name
                                    all_slot.update(frame['state']['slot_values'])

                                    expression = Expression(turn['utterance'], frame['state']['active_intent'], slots,frame['service'],string_list,vague_slot_names,dialogue['dialogue_id'])
                                    expressions.append(expression)
                                    if frame['state']['active_intent'] not in intent_template_dict.keys():
                                        intent_template_dict[frame['state']['active_intent']]=IntentTemplate(frame['service'])
                                    intent_template_dict[frame['state']['active_intent']].add_sample(expression)

                            pre_slot_name=all_frame_slot_name
                            pre_slot=all_slot

    for item in intent_template_dict.values():
        item.slot_val_statics()
    return expressions,intent_template_dict


class SlotExample:
    def __init__(self, quintuple):
        self.type = "slot"
        self.intent = quintuple[0]
        self.sentence = quintuple[1]
        self.slot = quintuple[2]
        self.flag = quintuple[3]
        self.span = quintuple[4]

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class SlotExampleGenerator:
    """
    generate examples
    """
    def __init__(self, training_percentage, neg_percentage, seed = None):
        if training_percentage < 0.0 or training_percentage > 1.0:
            raise ValueError("training_percentage is out of range")
        self.neg_percentage = neg_percentage
        self.training_percentage = training_percentage
        self.seed = seed
        self.all_sample=[]
        self.used_template=set()

    def __call__(self, expressions, slots_dict,intent_template_dict):
        examples = defaultdict(list)
        random.seed(self.seed)
        total_pos_cnt=0

        total_neg_cnt=0
        for expression in expressions:
            # print("generate expression")
            intent = expression.intent
            utterance = expression.utterance
            vague_slot_names=expression.vague_slot_names
            print(f'  expression.id :{expression.id}  utterance: {utterance}   template: {expression.template}   expression.slots :  {expression.slots}  vague_slot_names  :{vague_slot_names}')
            if expression.template in self.used_template:
                continue
            self.used_template.add(expression.template)
            pos_slots = len(expression.slots)
            neg_slots = len(slots_dict[intent]) - pos_slots
            service_intent=expression.service+'.'+intent
            pos_count = 0
            for slot in slots_dict[intent]:
                if slot in expression.slots:
                    for slot_val,span in expression.slots[slot].items():
                        pos_count += 2
                        equal_sent,(new_start,new_end)=intent_template_dict[expression.intent].generate_equal_sample_fixed(expression,slot,slot_val)
                        examp = [service_intent,equal_sent.lower(), " ".join(slot.split('_')), "1", str(new_start)+", "+str(new_end)]
                        partition = dataset_type(FLAGS.training_percentage, FLAGS.dev_percentage)
                        examples[partition].append(json.dumps(SlotExample(examp).toJSON(), indent=4))
                        mask_len=random.randint(intent_template_dict[intent].val_len_range[slot][0],intent_template_dict[intent].val_len_range[slot][1])
                        assert   mask_len != 0
                        equal_sent,(new_start,new_end)=intent_template_dict[expression.intent].generate_equal_sample_fixed(expression,slot,slot_val,mask_len)
                        equal_sent=equal_sent.lower()
                        equal_sent=equal_sent.replace('[mask]','[MASK]')
                        examp = [service_intent, equal_sent, " ".join(slot.split('_')), "1", str(new_start) + ", " + str(new_end)]
                        partition = dataset_type(FLAGS.training_percentage, FLAGS.dev_percentage)
                        examples[partition].append(json.dumps(SlotExample(examp).toJSON(), indent=4))

            neg_count=0
            for slot in slots_dict[intent]:
                if slot not in expression.slots  and slot not in vague_slot_names:
                    equal_sent=intent_template_dict[expression.intent].generate_utterance(expression)
                    examp = [service_intent, equal_sent.lower(), " ".join(slot.split('_')), "0", str((0, 0))[1:-1]]
                    partition = dataset_type(FLAGS.training_percentage, FLAGS.dev_percentage)
                    examples[partition].append(json.dumps(SlotExample(examp).toJSON(), indent=4))
                    neg_count+=1
            total_pos_cnt+=pos_count
            total_neg_cnt+=neg_count
        print(f'total_pos_cnt:{total_pos_cnt}  total_neg_cnt:{total_neg_cnt}  total_cnt:{total_pos_cnt+total_neg_cnt}  ')
        return examples

def save(slot_examples, path):
    """
    save generated examples into tsv files
    :param slot_examples:  generated examples
    :param path: output path
    :return: None
    """
    for key, examples in slot_examples.items():
        # key is train or test
        with open(os.path.join(path, MODEL+'ADD_MASK_FRANE_CUT.'+FLAGS.input[2:-1]),'w') as f:
            for example in examples:
                f.write(example+"\n")
    return

def main(_):
    SLOTS = generate_slotindex(FLAGS.input)

    slot_expressions,intent_template_dict = load_add_slot_value(FLAGS.input,SLOTS)
    build_slot_examples = SlotExampleGenerator(FLAGS.training_percentage, FLAGS.negative_proportions)
    if FLAGS.fix_random_seed:
        build_slot_examples.seed = 202006171752

    slot_examples = build_slot_examples(slot_expressions, SLOTS, intent_template_dict)
    save(slot_examples, FLAGS.output)

if __name__ == '__main__':
    app.run(main)
