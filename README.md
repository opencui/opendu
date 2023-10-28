# LUG, Language Understanding and Generation

For now, this repository contained retrieval augmented generation based implementation of language understanding (
a term commonly use in chatbot development community), or function calling known tool-using. 

It is designed to be an open source implementation of function calling, which can be used with any LLM with
provided finetune implementation for both embedding model and generation model. We will also make efficient inference
possible based on excellent project like llama.cpp, vllm, for example. 


There are couple basic goals for this repo:
1. It should use the same return as OpenAI function calling API.
2. It can be instructed by OpenAPI function specifications.
3. It should be easy to fix understanding issues, with exemplars defined in OpenCUI format.
4. It should be easy to utilize the external entity recognizer for slot filling. 


## How do you define the conversation?


## Model supported
The main hypothesis is that functional calling does not need LLM with extreme large parameters. The model
we based our fine-tuning on is all small models, so it is easy to deploy.
- [] [TinyLlama-1.1B](https://github.com/jzhang38/TinyLlama)

## Example Usage


```opencui
from opencui import Structifier, FunctionId, IntentLabel, EntitySpans, SlotMeta, SlotValue

// Setup Structifier.
structifier = Structifier(huggingface_model_path)
// One can add more than one specs.
structifier.addSpecs(api_specs)
// One can add more than one spaces.
structifier.addExemplars(exemplars)
// This should persist to disk. 
structifier.finalize(path_to_persist).


// Inference:
structifier = Structifier.loadFromDisk(path_to_persist)
structifier.convert(utterance)
```

## Acknowledgements
This project is relying on many impactful open source projects, we list the most important ones:
1. [LlamaIndex](https://github.com/run-llama/llama_index) for RAG implementation.
2. [TinyLlama-1.1B](https://github.com/jzhang38/TinyLlama) for tiny llama model.
3. [huggingface.ai](https://huggingface.ai) for dataset, fine-tuning
