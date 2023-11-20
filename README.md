# LUG 

LUG, short for Language Understanding and Generation, is an open source, retrieval augmented generation
(RAG) based function calling API implementation. It is designed both for dialog understanding in chatbot development
and tool-using agent development. Because of this, the terminology can be a bit confusing, we will use function
interchangeably with skill, intent, and parameter with slot. 

It can be used with any LLMs with provided finetune script for both embedding model and generation model.
Efficient inference is possible using excellent project like llama.cpp, vllm. With open sourced LLM, you 
can privately deploy the entire function calling API solution anywhere you want.

There are couple basic goals for this project:
1. It should use the same return as OpenAI function calling API.
2. It can be instructed by OpenAPI/OpenAI function schemas.
3. It should be easy to fix understanding issues, with exemplars defined in OpenCUI format.
4. It should be easy to utilize the external entity recognizer for slot filling. 

## What signal can be used to define conversion?
LUG takes three kind of different signal to shape how conversion is done:
1. Function schema, particular [OpenAPI](https://spec.openapis.org/oas/latest.html)/[OpenAI](https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions) function schema.
2. Function exemplar, the utterance template that is associated with function.
3. Entity recognizer for the slots.

### Function schema
For each function that you want to convert from natural language into structured representation, we first need its
schema, which include:
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
used during the training of LLMs, so it can be difficult to LLMs to recognize these entities out of box. LUG allow you
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
For now, the embedding model will be SentenceTransformer based.

For generation, the main hypothesis is that functional calling does not need LLM with extreme large parameters.
The model we based our fine-tuning on will be small models mostly, so it is easy to deploy. But there is no
reason that you can not use larger model. Since it is llama-index based, you should be able to use decoder model 
available via APIs.  

- [] [TinyLlama-1.1B](https://github.com/jzhang38/TinyLlama)

## Caveat
There are a couple of things that we plan to get to but there are not included in the initial version.
1. Currently, we only support conversion to functions from single module. The function in the single module are expected to be exclusive
in the semantic space.
2. We do not fully support overloaded functions yet, they are considered to be single function.
3. We do not support multiple functions mentioned in single utterance.
4. We do not support alternative semantics in parameter value, such as "No spice, please."
5. We do not pay special attention to implicature calculation yet. 

## Usage

To use converter, you just follow a simple three steps:

#### 1. Prepare the signal needed to define the conversation, namely:
1. Function schema, which can be in OpenAPI or OpenAI format. You can check the corresponding documentation for how to create these
schema.
2. Function exemplars, which should be mainly used to list the hard-to-understand utterances. It is important to provide the template instead of raw utterance there.
3. Entity recognizers, which should be used to help LLM to extract business dependent entities. 

#### 2. Build index.
More concretely, we assume that there will be three files in one directory: schema.json, exemplars.json and recognizers.json, 
one for each kind of signals defined above. 
```bash
python3 converter/lug_index.py -i <directory_you_read_specs_from> -o <directory_you_write_index_to>
```
By default, the code will not check the format for the function speciation, but it will check the format for the second
and third input and raise exception when there is issues.

#### 3. Initialize converter and convert.

```python
from converter.client
import Converter
converter = load_converter(directory_you_read_specs_from, directory_you_write_index_to)
converter.understand(utterance)
```

## Acknowledgements
This project is relying on many impactful open source projects, we list the most important ones:
1. [LlamaIndex](https://github.com/run-llama/llama_index) for RAG implementation.
2. [TinyLlama-1.1B](https://github.com/jzhang38/TinyLlama) for tiny llama model.
3. [huggingface.ai](https://huggingface.ai) for dataset, fine-tuning
