# LUG (language understanding and generation)

For now, LUG is an open source implementation of retrieval augmented generation based unction calling API
that is designed for dialog understanding in chatbot development and tool-using agent development. 

It is designed to be used with any LLMs with provided finetune script for both embedding model and generation model.
We will also make efficient inference possible based on excellent project like llama.cpp, vllm, for example. With open
sourced LLM, you can privately deploy the entire function calling API solution anywhere you want.

There are couple basic goals for this repo:
1. It should use the same return as OpenAI function calling API.
2. It can be instructed by OpenAPI function specifications.
3. It should be easy to fix understanding issues, with exemplars defined in OpenCUI format.
4. It should be easy to utilize the external entity recognizer for slot filling. 

## What signal do you use to define conversion?
Ralug takes three kind of different signal to shape how conversion is done:
1. Function specification, particular [OpenAPI](https://spec.openapis.org/oas/latest.html)/[OpenAI](https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions) function specification.
2. Function exemplar, the utterance template that is associated with function.
3. Entity recognizer for the slots.

### Function specification
For each function that you want to convert from natural language into structured representation, we first need its
specification, which include:
- description: A description of what the function does, used by the model to choose when and how to call the function.
- name: The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
- parameters: The parameters the functions accepts, described as a JSON Schema object. See the guide for examples, and the JSON Schema reference for documentation about the format.
OpenAPI and OpenAI have slightly different way to specify function.

An example in OpenAI format is as follows:
```json
{
    "name": "get_current_weather",
    "description": "Get the current weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the users location."
            }
        },
        "required": ["location", "format"],
    }
}
```
### Function exemplar
When certain user utterance is misunderstood, we need a quick and easy way to fix that. Under 
retrieval-augmented setting, this can be done by associating the utterance template, or exemplar, with a function.

An example in the json format is as follows:
```json
{
  "get_current_weather" : [
    {
      "template": "what is template in <location> in <format>?"
    }, 
    {
      "template": "How cold is there?"
    } 
  ]
}
```

### Entity recognizer for the slots.
It is common that business have some private entity that is not well represented in the public domain text that is
used during the training of LLMs, so it can be difficult to LLMs to recognize these entities out of box. RaLug allow you
to specify a recognizer for each slot, which recognizer can be a list of entity instance or some pattern.

An example in the json format is as follows:

```json
{
  "recognizers": {
    "city" : {
      "name": "city",
      "rec_type": "list",
      "description": "the place where people live",
      "instances": [
        {"label": "seattle", "expressions" :  ["seattle", "evergreen"]}
      ]
    }
  },
  "slots": {
    "location": "city"
  }
}
```

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
