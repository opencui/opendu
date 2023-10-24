<div align="center">
# Function Calling 
</div>

The repository contained retrieval augmented generation based implementation of function calling. Here, function calling
is responsible for converting natural text into function representation. It is essential for building chatbot and agent.

There are couple short term goal for OpenCUI function calling:
1. It is based on OpenAPI specification, thus it is easy to replace the OpenAI function calling API.
2. It should be easy to fix understanding errors, with exemplars.
3. It should be easy to utilize the external entity recognizer for slot filling. 

This project is intended to be a one-stop shop for function calling, it will come with fine-tuning for both
embedding model and LLM used for generation, as well as the efficient inference based on excellent project
like llama.cpp, vllm, for example.

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
