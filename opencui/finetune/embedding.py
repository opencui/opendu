import logging
from dataclasses import dataclass

from datasets import Dataset
from langchain.schema import BaseRetriever
from llama_index.schema import TextNode
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from opencui.core.annotation import FrameSchema
from opencui.core.config import LugConfig
from opencui.core.embedding import EmbeddingStore
from opencui.core.retriever import HybridRetriever
from opencui.finetune.commons import DatasetFactory


#
# This should be fully tested for fine-tuning embedding. Generally, there is no need for this
# since we do not really know what domain/topic we work on.
# For when we do, and when we need the extra improvement from the embedding, however small it is,
# we can try to run this.
# We might also want to rerun this for handling slot when we have enough of the pairs.
#
@dataclass
class DatasetCreatorWithIndex:
    creator: DatasetFactory
    desc_retriever: HybridRetriever
    exemplar_retriever: HybridRetriever

    @classmethod
    def build(cls, creator: DatasetFactory, path: str):
        return DatasetCreatorWithIndex(
            creator=creator,
            desc_retriever=HybridRetriever(path, "desc", LugConfig.get().desc_retrieve_topk),
            exemplar_retriever=HybridRetriever(
                path, "exemplar", LugConfig.get().exemplar_retrieve_topk
            ),
        )


def train(model: SentenceTransformer, dataset: Dataset, model_save_path: str):
    word_embedding_model = model._first_module()

    # We try to use these special tokens for potential roles.
    tokens = ["[DOC]", "[QRY]"]
    word_embedding_model._tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(
        len(word_embedding_model._tokenizer)
    )

    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(dataset, train_loss)],
        output_path=model_save_path,
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True,
    )


def create_sentence_pair_for_description(
        skills: dict[str, FrameSchema],
        dataset: Dataset,
        retriever: BaseRetriever,
        num_neg=1,
):
    embedding = EmbeddingStore.get_embedding_by_task("desc")
    results = []
    for item in dataset:
        utterance = item["utterance"]
        query = embedding.expand_for_query(utterance)
        label = item["owner"]

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
            node_label = node.metadata["owner"]
            if count < num_neg and label != node_label:
                count += 1
                results.append(InputExample(texts=[query, content], label=0.0))
    return results


def create_sentence_pair_for_exemplars(
        dataset: Dataset, retriever: BaseRetriever, num_examples=1
):
    embedding = EmbeddingStore.get_embedding_by_task("exemplar")
    results = []
    for item in dataset:
        utterance = item["utterance"]
        query = embedding.expand_for_query(utterance)
        label = item["owner"]
        id = item["id"]
        nodesWithScore = retriever.retrieve(utterance)
        nodes: list[TextNode] = [item.node for item in nodesWithScore]
        pos = 0
        neg = 0
        for node in nodes:
            content = embedding.expand_for_content(node.text)
            node_label = node.metadata["owner"]
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
        dataset = dataset_info.creator["train"]
        generators.extend(
            create_sentence_pair_for_description(
                dataset_info.creator.schema.skills, dataset, dataset_info.desc_retriever
            )
        )
        generators.extend(
            create_sentence_pair_for_exemplars(dataset, dataset_info.exemplar_retriever)
        )
    return generators


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    from opencui.finetune.commons import (
        DatasetCreatorWithIndex,
        has_no_intent, DatasetFactory,
    )

    print(LugConfig.get().embedding_model)
    dsc = [
        DatasetCreatorWithIndex.build(
            JsonDatasetFactory("./datasets/sgd/"),
            "./index/sgdskill/")
    ]
    dataset = DataLoader(generate_sentence_pairs(dsc))
    base_model = EmbeddingStore.get_model(LugConfig.get().embedding_model)
    train(base_model, dataset, "./output/embedding/")
