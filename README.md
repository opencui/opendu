# structifier
The repository for open source natural language understanding module for OpenCUI, which is responsible for converting
natural text into structured representation of semantics.

There are couple short term goal for OpenCUI NLU:
1. Structifier should be configured with OpenAPI specification.
2. It should be easy to fix understanding errors, with exemplars.
3. It should be easy to use the external help for slot filling. 


The OpenCUI Structifier API is defined as following:

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
