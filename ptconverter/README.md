OpenCUI Dialog Understanding

The OpenAI's new function calling API demonstrated that LLM can be used to convert text into semantic representation required for calling functions. However, there is a couple of fundamental issues with its design and made it impossible for production use:
1. It is very hard to fix understanding issue, other than reporting issues to OpenAI. 
2. Under multi-turn conversation, user utterance might not mention intent or function to be triggered at all. 
3. When there are new or business specific entities, there is no easy way for developer to make that information to help understanding.

Of course, there are other issues, like API might be off the table for the privacy considerations. OpenCUI DU is a LLM based dialog understanding that try to fix all the issues while keep the excellent developer experience afforded with zero shot capabilities offered by the LLMs. 

The conversion from text to json or other structure representation of user intention is a fundamental capability for conversational user interaction. And it is useful to have a standard interface and solution so that chatbot developer can focus on build business dependent interactions instead of implementation detail.

```json
{
  "source": "where the labeled example come from for debugging purpose",
  "kind": "description or exemplar, indicating how to use the label example",
  "utterance": "user input",
  "reference": "probe that we need to related utterance to",
  "label": "2/1/0, implies/means/none"
}
```

```json
{
  "source": "where the labeled example come from for debugging purpose",
  "utterance": "user input",
  "skill": "the label that identifies skill/intent/function",
  "candidates": [
    {
      "slot_label" : "the label for the slot",
      "type": "entity/frame, for now we only support entity",
      "start":  "character index, beginning of the candidate value, not always the true slot value",
      "end": "end of the candidate value, in character index" 
    },
    {}
  ],
  "label": [
    {
      "slot_label" : "the label for the slot",
      "type": "entity/frame, for now we only support entity",
      "start":  "character index, beginning of the slot value",
      "end": "end of the slot value, in character index" 
    },
    {}
  ]
}
```

The low level API from the raw model is the following
```openapi
typealias Kind description|exemplar
typealias Prediction implies|means|none
fun getIntentRaw(utterance: String, references: List<Pair<Kind, String>>): List<Prediction>

class SlotValue{
    val label: String
    val type: Entity (for now)
    val operator: 
    val start: Int
    val end: Int
    }
fun getSlotRaw(utterance: String, intent: String, candidate: Map<String, List<SlotValue>>): Map<String, List<SlotValue>>
```



In long term, in no particular order:
1. Figure out how to deal with multi intents in a single utterance. 
2. Handle operators: not, or, and special value [don't care]. 
3. Handle frame slot filling. 
4. Carry out implicature calculation.