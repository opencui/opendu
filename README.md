# nlu
The repository for open source natural language understanding forOpenCUI. The dialog understanding in OpenCUI consisits of low level
natural language understanding component, and then up level that turns the context independent natural language understanding into context
dependent dialog understanding, using the stack of frame as context.


There are two main goal for OpenCUI NLU:
1. It should be easy to fix understanding errors. 
2. It should be easy to use the external help, particularly with slot filling. 

In long term:
1. We need to figure out how to deal with multi intents in a single sentence. 
2. We need to handle operators: not, or, don't care. 
3. We need to handle composite type. 
4. We need to carry out implicature calculation.

