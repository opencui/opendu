# Finetuning for Dialog Understanding
  
The framework itself is designed to provide most developers with sufficient performance through default configurations and hotfixing capabilities out-of-the-box already. The fine-tuning support is to enable advanced developers to maximize the performance of this framework.

There are at least two instructed models you might want to fine-tune for better performance: embedding and generator. Both require a reasonable amount of labeled examples. We support two different approaches for fine-tuning the generator:

1. Full: Focusing on smaller LLMs.
2. LoRA: Allowing fine-tuning of larger models.

These approaches don't change the overall process, just how training and inference are set up. 

Note: Using only user utterances and slot values may not be sufficient to recover the corresponding exemplar, especially if a slot value appears multiple times in the utterance. To address this, the labeling process should:

1. Explicitly provide exemplars, or
2. Provide spans for each value

Failing to do so risks some labeled examples being discarded during fine-tuning.

Currently, we assume that a parameter/slot is recognized either by a recognizer or an LLM, but not both. This approach is adequate for tool-use use cases. However, for dialog understanding, there are some additional considerations depending on the state tracking implementation.

## How to finetune the generator
You must first define the schema. Once you have a schema, the main steps are:
1. Prepare raw labeled datesets in the form of AnnotatedExemplars.
2. Create the index for retrieval, for each module.
3. Create the fine-tuning dataset based on retrieval result, the prompt template and inference strategy.
4. Run fine-tuning code to create a fine-tuned model for inference.

### Prepare raw labeled examples
For now, we assume that the labeled examples for fine-tuning the generator require the following components:

1. User utterance 
   Example: "I'd like to get a ticket to New York"

2. Target name (function, intent, or skill name)
   Example: "buy_airline_ticket"

3. Target arguments (parameter or slot values)
   Example: {"quantity": 1, "destination": "new york"}

Note that we need utterance template, an normalized version of the original user utterance, for inference. With user utterance and slot values, we can recover corresponding exemplar for most of time. But when some slot value occurs more than one time in the utterance, we will run into trouble. So such case, the label process need to explicitly provider exemplars, or provide span for each value, or risk such labeled example is discarded for RAG inference. 

Per standard practices, it will be good if you prepare three datasets, for training, validation and testing. An example raw datasets is [schema-guided dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue). These raw dataset can be prepared in any format, as long as it also come with the code to generate the dataset in form of AnnotatedExemplars, of course, we also need to generate the information needed for RAG inference (as our fine-tuning dataset will be using the same prompt template as RAG).


### Create the index for retrieval
Given the schema and exemplars generated for each module, it's straightforward to create the index for retrieval, as this generates the dynamic context needed by the prompt template. However, special care must be taken to ensure that the instantiated prompt meets the criteria you had when designing the prompt template. 

For example, if you assume that the correct answer should be included in the prompt, you need to verify that this is indeed the case. Run experiments to determine the correct values for hyperparameters.

Use the following script to determine the hyperparameters for retrieval:

```bash
python3 finetune/find_k_for_prompt.py 
```

### Create the fine-tuning dataset
The actual fine-tuning dataset is created based on the following components:
1. Raw dataset
2. Retriever
3. Semantic parsing strategy
4. Chosen prompt template

Different semantic parsing strategies cast the dialog understanding task into different forms, thus requiring the generation of different fine-tuning datasets.

For some dialog understanding strategies, we need to generate negative examples. This requires that:
1. Functions from each module are mutually exclusive in the semantic space
2. There are no overloaded functions (these need to be handled in state tracking)

With these assumptions, we can select similar utterances from other functions as negative examples.

Finally, we can place training data we get for each module in the separate directories in coded in: <org>_<bot>_<lang>. 


### Fine-tune the generation model
Regardless we are using full fine-tuning or lora, we will focus on use one model for different tasks. The main reason is that dialog understanding is a form of translation from one representation to another, which is naturally a good fit for language models. We have two example scripts for you to get started.


## Decomposed modeling

For now, we decompose the dialog understaing into two steps: intent detection and slot value extraction. Basically, we first predict function name given user utterance, and then we predict argument values for given function and utterance. Since we are using decoder-only model, both tasks, along with other required tasks can be trained in a multitask fashion, in either full fine-tuning or lora based. So these models are really just different prompt templates based on the same model. 
 

### Intent dection/Function name prediction
For this sub-task, we have two approaches:
1. KNN based approach, where we fine-tune a model to predict whether a given utterance and a provided exemplar means the same, and the utterance will share the same function name with the exemplar.
2. RAG based approach, where we fine-tune a model to predict the function name given the descriptions fo the function, and example utterances and their labels.

### Slot value extraction

For this sub-task, we have two approaches:
1. Only extract value for one slot at each time,
2. Extract value for all slot in one shot. 

These RADU models will be trained on both public dataset, and the data generated from OpenCUI platform. We will potentially have two different models: specialized for one bot, and generalized for all bots. The first choice is useful for isolated the performance, and private deployment. The second choice is useful for dev environment on platform. Here we focus on how to do full fine-tuning for latter.


Reference:
1. [How to fine-tune both retriever and generator for better performance](https://arxiv.org/pdf/2310.01352.pdf)
2. [How to fine-tune retriever assuming blackbox generator](https://arxiv.org/pdf/2301.12652.pdf)

