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
2. Create the fine-tuning dataset based on the prompt template and inference strategy.
3. Run fine-tuning code to create a fine-tuned model for inference.

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


### Create the fine-tuning dataset
The actual fine-tuning dataset is created based on the raw dataset, semantic parsing strategy, together with prompt template of the choice. 

There are many valid semantic parsing strategy, depending on how we make use of language models. For example, we can use a 
To create the training examples from the supplied schemas and exemplars, it is possible that we need to generate
negative examples, this requires that the functions from each module is mutually exclusive in the semantic space,
and there is no overloaded functions. Overloaded functions need to handled in the state tracking. At beginning, we only
support the single module, but there is plan to support multiple modules.


### Determine the hyperparameters for retrieval
One of the main consideration for fine-tuning generation for RAG is what dynamic information are needed by prompt.
So special care need to be taken to make sure that instantiated prompt meet the criteria you had when you design the 
prompt templated. For example, if you assume that the correct answer should be included in the prompt, then you will
need to make sure that is indeed is the case. So you need to run some experiments to make sure that you choose the
correct value for hyperparameters.

The script you can use to determine the hyperparameter for retrieval:
```bash
python3 funetune/find_k_for_prompt.py 
```
Assuming that you have schema-guided dialogue dataset at <dir for lug>/../dstc8-schema-guided-dialogue/

### Fine-tune the generation model
By default, OpenLug uses a cascade model, where we use a skill model to determine the skill/function 
first, then use a different model for determine the slot/parameter values. Ideally, both models, along with other
required models are trained in a multitask fashion, in either full fine-tuning or lora based. It is also possible to
fine-tune requires generation models separately, but that is not what we aim for now. Mainly by changing the shared 
configuration in core/config.py and training aspect in finetune/generation.sh.  

The NLU models will be trained on both public dataset, and the data generated from OpenCUI platform. We will
potentially have two different models: specialized for one bot, and generalized for all bots. The first choice is
useful for isolated the performance, and private deployment. The second choice is useful for dev environment on
platform. Here we focus on how to do full fine-tuning for latter.

Since we are using a decomposed model, more directly pair wise discriminative model, for the hot fix capability,
the data for fine-tuning can be generated in two different ways: 
1. Labels are available before retrieval, so that we need to build index and produce retrieved pairs. 
2. Labels are available on pairs already so that we just need to add prompt to it.

We use decomposed discriminative model mainly because it is easy to provide labels, as operators only need to
fix the parts that need to be fixed instead of needing to provide the full label every time.



#### Prepare the for dataset.
Assuming that we put dataset in a directory called root. Then we can place training data we get for each bot in
the separate directories in coded in: org_bot_lang. 


### Testing
The follow python script can be used to test finetuned models.
```bash
python3 fineturn/test.py 
```


## Special considerations
Fine-tuning generator for RAG applications bring some new considerations. 

### How to retrieve
The description can be retrieved using embedding only. The exemplar should be retrieved in hybrid mode, with both
embedding and  keyword search. The vector search for description should use asymmetrical dual embedding, while for 
exemplar, vector search should use symmetrical embedding, using the same model for both user utterance and exemplars.

### How to model the conversion
There are many potential system architecture to model the conversation. For example, we can do one-shot model, or using
single model or prompt to predict both function name and its arguments. We can also have two models, one for predicting
function name, and one for predicting arguments. In this latter approach, we have two approaches:
1. Only extract value for one slot at each time,
2. Extract value for all slot in one shot. 

We will focus on using two models to solve the conversion, and focus on extracting value for one slot each time. Of
course, since we are using decoder-only model, two models are really just two different prompt templates typically
using the same model.


Reference:
1. [How to fine-tune both retriever and generator for better performance](https://arxiv.org/pdf/2310.01352.pdf)
2. [How to fine-tune retriever assuming blackbox generator](https://arxiv.org/pdf/2301.12652.pdf)

