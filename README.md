# RADU

RADU (Retrieval-Augmented Dialog Understanding) is an open-source implementation of semantic parsing based on retrieval-augmented generation (RAG) techniques. Since the goal of dialog understanding is to convert natural language into a structured representation for some function, we use the following terms interchangeably:
- 'function', 'skill', and 'intent'
- 'parameter' and 'slot'


It can be used with any LLMs with provided finetune script for both embedding model and generation model.
Efficient inference is possible using excellent project like llama.cpp, vllm. With open sourced LLM, you 
can deploy the entire dialog understanding or function calling API solution anywhere you want. For now,
we focus on decoder-only or encoder-decoder models required by text generation.

There are couple basic goals for this project:
1. It should produce the same return as OpenAI function calling API.
2. It can use both OpenAPI/OpenAI function schemas.
3. It should be easy to fix understanding issues, with exemplars defined in OpenCUI format.
4. It should be easy to take advantage of the external entity recognizer for slot filling. 

## What signal can be used to define conversion?
RADU takes three kinds of different signals to shape how conversion is done:
1. Function schema, particularly [OpenAPI](https://spec.openapis.org/oas/latest.html)/[OpenAI](https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions) function schemas.
2. Expression exemplars: utterance templates associated with specific functions, where the templates are designed to trigger their corresponding functions.
3. Entity recognizers: external entity recognizers that can extract value for the slots or function parameters from user utterance.

### Function schema
To convert natural language into a structured representation for each function, we first need its schema. This schema includes:

- **description**: A summary of the function's purpose, guiding the model on when and how to call it.
- **name**: The function's identifier. It must use only a-z, A-Z, 0-9, underscores, or dashes, with a maximum length of 64 characters.
- **parameters**: The inputs the function accepts, described as a JSON Schema object. Refer to the guide for examples and the JSON Schema reference for format details.

Note that OpenAPI and OpenAI have slightly different methods for specifying functions.

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
        "required": ["location", "format"]
    }
}
```

### Function exemplar
Function exemplars are utterance templates associated with specific functions. They provide a quick and efficient way to correct misunderstandings in user input within a retrieval-augmented setting. By linking these exemplars to functions, we can improve the system's ability to accurately interpret and respond to user utterances.

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

### Entity recognizer for the slots (Not implemented yet)
It is common for businesses to have private domain entities that are not well-represented in the public domain text used for training LLMs. This can make it difficult for LLMs to recognize these entities out of the box. RASP allows you to specify a recognizer for each slot, which can be either a list of entity instances or a pattern.

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
The embedding model will be SentenceTransformer based. We will update the code when stronger embedding models become available. For now, we support BAAI and Stella family of embedding models.

For generation, we assume that dialog understanding doesn't require extremely large parameter models. We focus on smaller LLMs that can be fine-tuned using consumer-grade GPUs, making deployment easier and more accessible. However, there's no restriction on using larger models if desired.

- [&#x2714;] [T5](https://huggingface.co/docs/transformers/model_doc/t5)
 
## Caveat
There are several features that we plan to implement but are not included in the initial version:

1. We support multiple modules (think of as a collection of functions), but the functions in the same module are expected to be mutually exclusive in the semantic space, a requirement for how we fine-tuning,
2. We will not support overloaded functions yet; they will be treated as a single function at semantic parsing level.
3. We do not support multiple functions mentioned in a single utterance yet.
4. We only support assignment semantics in parameter values, so no "No spice, please."
5. We do not yet pay special attention to implicature calculation.

## Usage
To use the converter, follow these three simple steps:

#### 1. Prepare the signals needed to define the conversation:
1. Function schema: Can be in OpenAPI or OpenAI format. Refer to the corresponding documentation for creating these schemas.
2. Function exemplars: Mainly used to list hard-to-understand utterances. Provide templates instead of raw utterances.
3. Entity recognizers: Used to help LLMs extract business-dependent entities.

#### 2. Build index
Place three files in one directory: `schema.json`, `exemplars.json`, and `recognizers.json`, one for each signal type defined above.

```bash
export PYTHONPATH="$PYTHONPATH:."
python3 opencui/inference/index.py -s <directory_you_read_schema_from>
```
By default, the code won't check the format for function specifications but will verify the format for exemplars and recognizers, raising exceptions if issues are found.

#### 3. Initialize converter and convert
Note:
You can also test it in the command line:

```bash
export PYTHONPATH="$PYTHONPATH:."
python3 inference/cmd.py -s <directory_you_read_schema_from>
```

## Acknowledgements
This project is relying on many impactful open source projects, we list the most important ones:
1. [LlamaIndex](https://github.com/run-llama/llama_index) for RAG implementation.
2. [huggingface.ai](https://huggingface.ai) for dataset, fine-tuning
3. [S-Lora](https://github.com/S-LoRA/S-LoRA) For efficient inference with multiple lora adaptors.
