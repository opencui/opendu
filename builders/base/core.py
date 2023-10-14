import os
from collections import defaultdict

import torch
from evaluate import load
import gin
import json
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import random
import time
import pandas as pd


@gin.configurable
class PerplexityCalculator:
    def __init__(self, model_id='gpt2'):
        self.perplexity = load("perplexity", module_type="metric")
        self.model_id = model_id

    def __call__(self, input_texts):
        print(input_texts)
        result = self.perplexity.compute(model_id=self.model_id, add_start_token=False, predictions=input_texts)
        print(result)
        return result["perplexities"]


def save(examples, path, label):
    """
    save generated examples into tsv files
    :param examples:  generated examples
    :param path: output path
    :return: None
    """
    # we only generate one file for each folder
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

    with open(os.path.join(path, label), 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(example.toJSON() + "\n")
    return


class IntentExample:
    def __init__(self, src, label, utterance, tokenized, exemplar=True):
        self.type = "intent"
        self.kind = "exemplar" if exemplar else "description"
        self.source = src
        self.label = label
        self.utterance = utterance
        self.exemplar = tokenized

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4).replace("\n", "")


@gin.configurable
class ModelEncoder:
    def __init__(self, model_ckpt):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model_ckpt
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).to(self.device)

    def convert(self, text_list):
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        encoded_input = {k: v for k, v in encoded_input.items()}
        return self.model(**encoded_input).last_hidden_state[:, 0]


def query(encoder, text, dataset, k=4):
    embedding = encoder.convert([text])[0].detach().cpu().numpy()
    scores, samples = dataset.get_nearest_examples("embeddings", embedding, k=k)
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["score"] = scores
    samples_df.sort_values("score", ascending=False, inplace=True)
    return samples_df


@gin.configurable
class GenerateIntentExamples:
    """
    generate examples from templates.
    """
    def __init__(self, encoder, training_percentage, negative_percentage, seed=None):
        if training_percentage < 0.0 or training_percentage > 1.0:
            raise ValueError("training_percentage is out of range")
        self.neg_percentage = negative_percentage
        self.training_percentage = training_percentage
        self.seed = seed
        self.desc_k = 8
        self.encoder = encoder
        self.pos_num = 16

    def __call__(self, templates, descriptions):
        examples = []
        random.seed(self.seed)
        starttime = time.time()

        for intent, meta in templates.items():
            # first create description related
            print(f"Handling {intent}")
            for exemplar, expression in meta.exemplars.items():
                utterance = meta.generate_utterance(expression)

                # Sample from description
                top_k_description = query(self.encoder, utterance, descriptions, k=8)
                for _, row in top_k_description.iterrows():
                    label = 1 if intent == row['source'] else 0
                    desc = row['text']
                    examples.append(IntentExample(intent, label, utterance, desc, False))

                # Sample from positive example
                pos_selection = random.sample(list(meta.exemplars.items()), self.pos_num)
                for lexemplar, lexpression in pos_selection:
                    if lexemplar == exemplar:
                        continue
                    tokenized = lexpression.tokenize_label()
                    examples.append(IntentExample(intent, 1, utterance, tokenized, True))

                # Sample from negative example:
                neg_examples = []
                for lintent, lmeta in templates.items():
                    if lintent == intent:
                        continue
                    top_k_negatives = query(self.encoder, utterance, lmeta.dataset, k=8)
                    for _, row in top_k_negatives.iterrows():
                        score = row['score']
                        tokenized = row["text"]
                        neg_examples.append((lintent, tokenized, score))
                neg_examples.sort(key=lambda x: x[2], reverse=True)
                for example in neg_examples[0:self.pos_num]:
                    examples.append(IntentExample(f"{intent}|{example[1]}", 0, utterance, example[1]))
        return examples


class SearchSimilarExpressions:
    """
    using sentence-transformer to encode all the utterance with new intent
    """

    def __init__(self, intent_expressions):
        self.expression_corpus = []  # expression corpus used to be encoded by bert for all expressions
        self.idx2expression = {}  # map idx to expression object
        self.intent_range = {}
        self.sentence_embeddings = None
        self.tfidf_matrix = None
        self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        idx = 0
        stt = 0
        for intent, expressions in intent_expressions.items():
            end = len(expressions)
            # give the range of the expressions in the expression_corpus for every intent,  left closed right open
            self.intent_range[intent] = (stt, stt + end)
            stt += end
            for expression in expressions:
                self.expression_corpus.append(expression.utterance)

                expression.idx = idx
                # given the index for the order of the expression,idx indicates the order of the
                # sentence in the total expressions
                self.idx2expression[idx] = expression
                idx += 1

        idf_vectorizer = TfidfVectorizer(use_idf=True)
        self.tfidf_matrix = idf_vectorizer.fit_transform(self.expression_corpus).toarray()
        self.sentence_embeddings = self.model.encode(self.expression_corpus)


def check_stop_words(slot_dict, utterance, string_list, stop_words_path):
    stop_words = []
    with open(stop_words_path, encoding='utf-8') as f:
        items = f.readlines()
        for t in items:
            stop_words.append(t.lower()[:-1])
    stop_words = set(stop_words)
    # print("stop_words",stop_words)
    single_dict = dict()
    if string_list:
        for key, values in slot_dict.items():
            for value in values:
                single_dict[value] = key
        string_list = sorted(string_list, key=lambda x: x[0])
        res_utterance = utterance[:string_list[0][0]]
        for i, (cur_start, cur_end) in enumerate(string_list):

            if i == len(string_list) - 1:
                res_utterance = res_utterance + utterance[cur_end:]
            else:
                res_utterance = res_utterance + utterance[cur_end:string_list[i + 1][0]]

    else:
        res_utterance = utterance
    import string
    punctuation_string = string.punctuation
    for i in punctuation_string:
        res_utterance = res_utterance.replace(i, '')

    all_not_slot_words = set(res_utterance.split())

    if len(all_not_slot_words - stop_words) >= 2:
        return True
    return False


def generate_expression_template(slot_dict, utterance, spans):
    '''
    replacing the slot val with the slot name,to avoid match the short slot val which may be included in other
    long slot val, we need sort by the length of the slot val
    '''
    if spans == []:
        return utterance
    single_dict = dict()

    for key, values in slot_dict.items():
        for value in values:
            single_dict[value] = key

    spans = sorted(spans, key=lambda x: x[0])
    res_utterance = utterance[:spans[0][0]]
    for i, (cur_start, cur_end) in enumerate(spans):
        # if len(string_list) >=2:
        #     print("sub string",utterance[cur_start:cur_end])
        res_utterance = res_utterance + ' < ' + single_dict[utterance[cur_start:cur_end]] + ' > '
        if i == len(spans) - 1:
            res_utterance = res_utterance + utterance[cur_end:]
        else:
            res_utterance = res_utterance + utterance[cur_end:spans[i + 1][0]]

    return res_utterance


class IntentMeta:
    """
    restore the all template of a certain intents, including the set of all possible exemplars,
    and the dict for all slot
    """

    def __init__(self):
        self.exemplars = dict()
        self.slot_dict = defaultdict(set)
        self.dataset = None

    def add_sample(self, expression):
        expression_template = generate_expression_template(expression.slots, expression.utterance, expression.spans)
        if expression_template in self.exemplars:
            return

        expression.exemplar = expression_template
        for slot_name, slot_val_list in expression.slots.items():
            for slot_val in slot_val_list:
                self.slot_dict[slot_name].add(slot_val)
        self.exemplars[expression_template] = expression

    def generate_utterance(self, expression):
        expression_template = generate_expression_template(expression.slots, expression.utterance, expression.spans)
        for slot_name, slot_vals in self.slot_dict.items():
            if '< ' + slot_name + ' >' in expression_template:
                expression_template = expression_template.replace('< ' + slot_name + ' >', list(slot_vals)[random.randint(0, len(slot_vals) - 1)])
        return expression_template

    def finalize(self, encoder):
        source = []
        exemplars_list = []
        for exemplar, expression in self.exemplars.items():
            source.append(exemplar)
            exemplars_list.append(expression.tokenize_label())

        results = pd.DataFrame({"exemplar": source, "text": exemplars_list})
        results = Dataset.from_pandas(results)
        results = results.map(
            lambda x: {"embeddings": torch.nn.functional.normalize(encoder.convert([x["text"]])).detach().cpu().numpy()[0]}
        )

        results.add_faiss_index(column="embeddings")
        self.dataset = results


class Expression:
    """
    expression examples
    """

    def __init__(self, expression, intent, slots, string_list=None):
        self.utterance = expression
        self.intent = intent
        self.slots = slots  # dict to store slot, value pairs
        self.idx = None
        self.spans = string_list
        self.exemplar = None
        self.service = None
        self.vague_slot_names = None

    def tokenize_label(self):
        no_underscore_utterance = self.exemplar
        for key, values in self.slots.items():
            no_underscore_utterance = no_underscore_utterance.replace(key, ' '.join(key.split('_')))
        return no_underscore_utterance

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)

def cover_reaction(expression_A, expression_B):
    '''
    check if the slot of A could cover all slot of B
    '''
    return set(expression_B.slots.keys()).issubset(set(expression_A.slots.keys()))


def slot_val_to_slot_name(slot_dict, utterance):
    '''
    replacing the slot val with the slot name,to avoid match the short val which may be included
    in other long val,we need sort by the length of the slot val
    '''
    single_dict = dict()

    for key, values in slot_dict.items():
        for value in values:
            single_dict[value] = key

    single_dict = sorted(single_dict.items(), key=lambda x: len(x[0]), reverse=True)

    for (value, key) in single_dict:
        utterance = utterance.replace(value, '< ' + ' '.join(key.split('_')) + ' >')

    return utterance

if __name__ == "__main__":
    calculate_perplexity = PerplexityCalculator()
    input_texts = [
        "i am still waiting on my card?",
        "what can i do if my card still hasn't arrived after 2 weeks?",
        "i have been waiting over a week. is the card still coming?",
        "can i track my card while it is in the process of delivery?",
        "how do i know if i will get my card, or if it is lost?",
        "where is my card?",
    ]

    res = calculate_perplexity(input_texts)
    print(res)
