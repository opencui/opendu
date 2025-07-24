# OpenDU

OpenDU, is an open source dialogue understanding module, that is an agentic Retrieval-Augmented Generation
(agentic RAG) based implementation for semantic parsing, it is designed to parse natural language into a structured
representation for semantic. In addition to dialog understanding for chatbot development, this package can
potentially also be used for function calling for agent development. As such, we use the following terms interchangeably:

- 'function', 'skill', and 'intent'
- 'parameter' and 'slot'
- 'semantic parsing', 'function calling' and '(dialog) understanding'

It can be used with any LLMs with provided finetune script for both embedding model and generation models. Efficient
inference is possible using excellent project like llama.cpp, vllm. With open sourced LLMs, you can deploy the
entire dialog understanding or function calling API solution anywhere you want. We focus on decoder-only or
encoder-decoder models required by text generation.

There are couple basic goals for this project:

1. It should produce the same return as OpenAI function calling API.
2. It can use both OpenAPI/OpenAI function schemas.
3. It should be easy to fix understanding issues, with exemplars defined in OpenCUI format.
4. It should be easy to take advantage of the external entity recognizer for slot filling.

## What signal can be used to define conversion?

OpenDU takes three kinds of different signals to shape how conversion is done:

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
  "get_current_weather": [
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
    "city": {
      "name": "city",
      "rec_type": "list",
      "description": "the place where people live",
      "instances": [
        { "label": "seattle", "expressions": ["seattle", "evergreen"] }
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

To use the converter, you first need to prepare the inputs needed to define the semantic parsing:

1. Function schema: Can be in OpenAPI or OpenAI format. Refer to the corresponding documentation for creating these schemas.
2. Function exemplars: Mainly used to list hard-to-understand utterances. Provide templates instead of raw utterances.
3. Entity recognizers: Used to help LLMs extract business-dependent entities.

### Command line mode

Place three files in one directory: `schema.json`, `exemplars.json`, and `recognizers.json`, one for each input type defined above.

```bash
python3 opendu/inference/index.py -s <directory_you_read_schema_from>
```

By default, the code won't check the format for function specifications but will verify the format for exemplars and recognizers, raising exceptions if issues are found.

You can now test it in the command line:

```bash
python3 inference/cmd_test.py -s <directory_you_read_schema_from>
```

### Service Mode

The entire process can also be accessed via a restful end point. To start the service, simply:

```bash
python3 opendu/inference/service.py -s examples/
```

To make in the index:

```bash
curl http://<host>:3001/v1/index/<directory_of_the_module>
```

To parse the user utterance:

```bash
curl -d '{"utterance" : "<your_input sentence>", "mode" : "SKILL"}' -H "Content-Type: application/json" -X post http://<host>:3001/v1/predict/<directory_of_module>
```

Todo: more detailed documentation for service.

## Special considerations

### How to retrieve

Retrieval methods differ for descriptions and exemplars:

1. Descriptions:

   - Use embedding-only retrieval
   - Employ asymmetrical dual embedding for vector search

2. Exemplars:

   - Use hybrid retrieval (combining embedding and keyword search)
   - Use symmetrical embedding for vector search
   - Apply the same model for both user utterances and exemplars

### Dialog understanding strategy

Given the function schema, exemplars, and entity recognizers, we can define multiple strategies to convert natural language into a structured representation for each function. For example, after retrieval:

1. We can use a function calling model to directly convert the user utterance into a structured representation. The main problem with this approach is that currently, these models only support function description as input, not examples in the input, so example based in-context learning is not possible.
2. Alternatively, we can first determine the function itself, and then generate the parameter values. Each of these sub-steps can also be solved using different strategies. For instance, slot filling can be approached as a set of question-answering problems, with one question for each slot and the user utterance as the passage.

We will support multiple strategies, allowing you to swap different components in and out to best fit your use case.

## Acknowledgements

This project is relying on many impactful open source projects, we list the most important ones:

1. [LlamaIndex](https://github.com/run-llama/llama_index) for RAG implementation.
2. [huggingface.ai](https://huggingface.ai) for dataset, fine-tuning
