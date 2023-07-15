# nlu
The repository for open source natural language understanding for OpenCUI. The dialog understanding in OpenCUI consists of low level
natural language understanding component, and then up level that turns the context independent natural language understanding into context
dependent dialog understanding.

There are couple short term goal for OpenCUI NLU:
1. Converter should be configured with OpenAPI specification.
2. It should be easy to fix understanding errors, with exemplars.
3. It should be easy to use the external help using entity slot filling. 


The OpenCUI NLU Converter API is defined as following:

```opencui
from opencui import Converter, FunctionId, IntentLabel, EntitySpans, SlotMeta, SlotValue

// Setup converter.
converter= Converter(huggingface_model_path)
// One can add more than one specs.
converter.addSpecs(api_specs)
// One can add more than one spaces.
converter.addExemplars(exemplars)
// This should persist to disk. 
converter.finalize(path_to_persist).


// Inference:
converter = Converter.loadFromDisk(path_to_persist)

# this returns List<Pair<IntentLabel, FunctionId>
candidate_intents = converter.toIntent(utterance: String)

# this returns  List<SlotValue>
candidate_slots = converter.toEntitySlot(utterance: String, funcId: FunctionId, potential_entities: EntitySpans)
```
