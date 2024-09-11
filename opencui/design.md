# RAU Design Notes 

Semantic parsing is a crucial step in converting natural language into structured representations for programmatic use. It can be done end to end using function calling capable models, but there are a couple of reasons function callign API is not ideal for semantic parsing for a couple of reasons:
- As of now, using function calling models directly requires we feed all function descriptions directly into API calls every time, this can be cumbersome for large applications with many functions, both in terms of time and cost.
- Function calling models are not yet very good at semantic parsing, so for complex tasks, they sometimes fail to produce the correct output. When that happens, it is hard to fix because it does not yet support exmaple based in-context learning.
- When there are priviate domain or new entities that foundation models are not aware of, it is hard to help these function calling models improve because fine-tuning them effectively requires a considerable effort.

To address the first issue, we can use a retrieval augumentation approach, dynamically pick the functions that might be relevant given the user's utterance, and only feed these function descriptions to API call,  so we do not have to feed all function descriptions into API calls every time. This save time and cost, and potentially improve the performance as well as it reduce the confusion if retrieval done well. To use example-based in-context learning to help the function calling model improve its semantic parsing ability, unfortunately, we have to fine-tune a model with in-context support, there is no getting around it.

While it is possible to fine-tune a one stage model, that goes from utterance directly to it semantic representation, we will explore a two stage strategy here: first determine which function (functions) is user utterance related, and then given a function, extract the parameter for that function from user utterance. This two stage strategy is essentially for the dialog understanding use case, as it is possible in a multi-turn conversation, the user's utterance does not trigger any function, just provide values for some parameters. (To support multi-turn conversation, we also requires involvement from dialog management, or interaction logic, to deal with missing information, but that is out of scope for this project.)

## Intent Detection
There are many different ways to do intent detection, we can cast it as a text classification task for example. In old day, intent detection is typically addressed as a supervised learning problem, which requries labeling a reasonable amout of data to train a model to predict the intent of a user's utterance. This can be a time consuming and expensive, thus not really a production friendly process. With the rise of foundation models, we have the opportunity to address intent detection as a few-shot or even zero shot learning problem, benefiting from the large amount of data that foundation models have been trained on. One of the key design goal is to be able to using new example to improve performance without always resolved to fine-tuning, so we are exploring the following two approaches:

### Non-parametric Classification

Non-parametric classification, also known as the nearest neighbor method, uses labels from the nearest training examples to make predictions. This approach can be applied to intent detection as follows:

1. Pick an LLM (potentially fine-tuned) to determine if a user utterance has the same meaning as an example utterance template.
2. Feed the model with a set of examples retrieved from a labeled dataset.
3. Select the function with the highest number of matching examples to the user utterance.

This approach is simple and efficient, and it can work well with a small LLM as the text equivalence checking is not a very complex reasoning task, and we expect the generalization performance to be good based on the context dependent representation of text. This text equivalence task is commonly included in the embedding and cross embedding training, except we treat this as a dedicated task. Clearly the semantic similarity between function description and user utterance is also important, and we can include that also in out nearest neighbor decision.

### In-Context Learning Approach

Another more holistic approach is to use one model to take in all the evidence, including both the function descriptions and example utterances along with implied functions. We can simply fine tune a small LLM, and craft an instruction that take all these information can generate the function name as output. This way, we can take advantage of the in-context learning capability of foundation models, and provide some hotfix capability to make the solution more production friendly. 

## Slot Filling
The slot filling problem is to extract the values for the parameters of given function. And there are many different ways to tackle this problem, we will explore the following LLM based approaches:

### Question Answering Approach
One of the most natural approach using LLM is cast this as a question answering problem, where the user utterance is the context, query is automatically constructed based on the function parameters, together they can be captured in one prompt, for example, "what is the value for <destination> in the following user utterance: '<i>I want to go to Paris</i>'? The answer is:", and hopefully LLM will complete with "Paris". Question answering is well studied problem with LLM, so we benetfit from all the related research. But there are some known issues: when there are multiple parameters, extracting values independently can cause confusion, for example, in "I want to go to Paris and New York", is "Paris" the destination or the departure? And it gets worse when there are nested parameters. 

### Information Extraction Approach
Structured information extraction is another well studied problem with significant research efforts, given an piece of input text and target schema, potentially nested, the goal is to extract values for all the slots simutaneously. One of the most effective method is to use a decoder only model with appropriate prompt, and train it with a large amount of annotated data. The main advantage is that it can reason about the parameters dependencies internally, and does not require any additional post processing to extract values. However, the performance is still not on par with the state of the art question answering based approach.

## Design Consideration
Based on above analysis, we assume that we will use a retrieval augmentation approach so that we do not need to feed all the functions to API all the time, we also focus on a two stage semantic parsing problem, so we can naturally support multi-turn conversation. In addition the example based in-context learning is also highly desired for its production friendness. For simpler function calling use case wehre both intention and parameters values are always mentioned in each turn, it is possible to provide a wrapper to combine both intent detection and slot filling into one API call.

### Index and retrieval
For exemplar search, We will focus on hybrid retrieval approach, using dense embedding for semantic similarity search, and mixture of k-nearest neighbor for exact string matching. For description based search, we will use dense embedding only, as we do not expect the function description overlap in words with user utterance that much. 

### Prompt design
One of the hallmark of instruction tuned foundation models is its ability of follow instrucitons. Well crafted instructions or prompt, can worth upwards of hundreds of labeled examples. So design good prompts are key to the success of LLM based solutions. We focus on structured prompts, which is generally considered to be good for small LLM models, and we will use the following structure. 

### Inference and Fine-tuning
We will separate the inference and fine-tuning to two different subdirectories. For each of them, we will support the interchageable use of different implementation of intent detection and slot filling solutions.  

References:
1. [How Many Data Points is a Prompt Worth?](https://arxiv.org/abs/2103.08493)