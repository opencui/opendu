import logging

from langchain.schema import BaseRetriever
from llama_index.schema import TextNode
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from finetune.datasets import Dataset
from torch.utils.data import DataLoader

from finetune.commons import SkillSpec, LugConfig
from core.embedding import EmbeddingStore
from core.retriever import DatasetCreatorWithIndex, has_no_intent


# There are many different flavor of embedding we need to support to chatbot building
# For example:
# 1. To support matching user utterance with function description, we need a pair of different embedding
# 2. To support matching user utterance with templated exemplars, we need a pair of difference embedding.
# 3. To support matching user utterance with spelled out exemplars, we need a pair of same embedding.
# 4. To support matching user utterance with answers, we need a pair of different embedding.
#
# To use the same model to create different embedding, the embedding can use with a pair of instructions.


def train(model: SentenceTransformer, dataset: Dataset, model_save_path: str):
    word_embedding_model = model._first_module()

    # We try to use these special tokens for potential roles.
    tokens = ["[DOC]", "[QRY]"]
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(dataset, train_loss)],
        output_path=model_save_path,
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True)


def create_sentence_pair_for_description(skills: dict[str, SkillSpec], dataset: Dataset, retriever: BaseRetriever, num_neg=1):
    embedding = EmbeddingStore.get_embedding_by_task("desc")
    results = []
    for item in dataset:
        utterance = item["utterance"]
        query = embedding.expand_for_query(utterance)
        label = item["target_intent"]

        # For random utterance, we do not have information to use it as anchor.
        if has_no_intent(label):
            continue

        nodesWithScore = retriever.retrieve(utterance)
        nodes: list[TextNode] = [item.node for item in nodesWithScore]

        content = embedding.expand_for_content(skills[label]["description"])
        results.append(InputExample(texts=[query, content], label=1.0))
        count = 0
        for node in nodes:
            content = embedding.expand_for_content(node.text)
            node_label = node.metadata["target_intent"]
            if count < num_neg and label != node_label:
                count += 1
                results.append(InputExample(texts=[query, content], label=0.0))
    return results


def create_sentence_pair_for_exemplars(dataset: Dataset, retriever: BaseRetriever, num_examples=1):
    embedding = EmbeddingStore.get_embedding_by_task("exemplar")
    results = []
    for item in dataset:
        utterance = item["utterance"]
        query = embedding.expand_for_query(utterance)
        label = item["target_intent"]
        id = item["id"]
        nodesWithScore = retriever.retrieve(utterance)
        nodes : list[TextNode] = [item.node for item in nodesWithScore]
        pos = 0
        neg = 0
        for node in nodes:
            content = embedding.expand_for_content(node.text)
            node_label = node.metadata["target_intent"]
            if pos == num_examples and neg == num_examples:
                break
            # We should never create pair with identical
            if id == node.id_:
                continue

            if node_label == label:
                if pos < num_examples:
                    pos += 1
                    results.append(InputExample(texts=[query, content], label=1.0))
            else:
                if neg < num_examples:
                    neg += 1
                    results.append(InputExample(texts=[query, content], label=0.0))
    return results


def generate_sentence_pairs(dataset_infos: list[DatasetCreatorWithIndex]) -> Dataset:
    generators = []
    for dataset_info in dataset_infos:
        dataset = dataset_info.creator.build("train")
        generators.extend(
            create_sentence_pair_for_description(
                dataset_info.creator.domain.skills,
                dataset,
                dataset_info.desc_retriever
            ))
        generators.extend(
           create_sentence_pair_for_exemplars(
                dataset,
                dataset_info.exemplar_retriever
            ))
    return generators


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    from finetune.datasets import SGDSkills
    print(LugConfig.embedding_model)
    dsc = [DatasetCreatorWithIndex.build(SGDSkills("/home/sean/src/dstc8-schema-guided-dialogue/"), "./index/sgdskill/")]
    dataset = DataLoader(generate_sentence_pairs(dsc))
    base_model = EmbeddingStore.get_model(LugConfig.embedding_model)
    train(base_model, dataset, "./output/embedding/")