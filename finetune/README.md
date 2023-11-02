# Finetuning for Language Understanding and Generation

There are at lease two instructed models you might need to fine-tune for better performance. But you can only
do this when you have some reasonable amount of labeled example available. The goal of fine-tuning code base
is make it possible for the advanced developers to squeeze the last bit of the performance of this architecture, 
for most of the developer, we hope the default configuration plus the hotfixing capability is enough.  

While it is possible to fine-tune both the generator and embedding needed by retrieval, we will support the 
fine-tuning in three steps:
1. Finetune generator, in full.
2. Finetune generator, using lora (this does not change overall process, just how to set up training and inference).
3. Finetune embedding.

Here support means we will have tested code and enough documentation for you to replicate the process. Additionally,
we will support the tool-use or function calling use case first, since dialog understanding related tasks are 
largely depending on the dialog state tracking design, different designs can have different requirements. We will
focus on the OpenCUI state tracking design for now. As in general, we only focus on the understanding related 
fine-tune for now. 

## How to finetune the generator for understanding

Fine-tuning the generator is an extra step needed for understanding. One still need to define the schema first. When
you have a schema, the main thing you need to do is: 
1. Prepare the labeled examples in form of FullExemplar.
2. Trigger the fine-tuning for generator, so that you can use the model for inference.

For now, we assume that the labeled examples in form of Expression for generator fine-tuning requires a couple of things:
- user utterance, for example: "I like to get a ticket to New York"
- exemplar, for templated utterance, for example: "I like to get <quantity> ticket to <destination>".
- target name, function (also known as intent, skill) name, for example "buy_airline_ticket"
- target value, parameter (or slot) values, for example: {"quantity" : 1, "destination" : "new york"} 

Notice, with just user utterance and slot values, we might not be able to recover corresponding exemplar, if some
slot value occurs more than one time in the utterance. So in general, the label process need to explicitly provider
exemplars, or provide span for each value, or risk some labeled example is discarded during the fine-tuning. For now,
we assume that a parameter/slot is recognized is either by recognizer or LLM, but not both. This requirement is only 
good enough for tool-use use cases, for dialog understanding, there are some additional considerations depending on the
state tracking implementation.

Per standard practices, it will be good if you prepare three datasets, for training, validation and testing. The first
datasets that we are using is based on [schema-guided dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).



Reference:
1. [How to fine-tune both retriever and generator for better performance](https://arxiv.org/pdf/2310.01352.pdf)
2. [How to fine-tune retriever assuming blackbox generator](https://arxiv.org/pdf/2301.12652.pdf)

