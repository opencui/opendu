#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
from pybars import Compiler


#
# We will use eos: </s> automatically in both train and decode. Prompt can decide whether
# and how they want to use bos: <s>.
# We need to have two path: one for training (we need extra_tokens) and one for decoding.
# In LUG, we call prompt needed by embedding instruction, as they are static. Templated prompt
# needed by generation will be called as prompt.
#
class ObjectLister:
    def __init__(
            self,
            item_header=None,
            header_delim: str = "\n",
            item_delim: str = "\n\n",
            block_header: str = "",
            block_tail: str = "",
            with_index: bool = True):
        self.item_header = item_header
        self.header_delim = header_delim
        self.item_delim = item_delim
        self.block_header = block_header
        self.block_tail = block_tail
        self.with_index = with_index

    def __call__(self, this, options, items):
        result = []
        result.append(self.block_header)
        for index, thing in enumerate(items):
            if index != 0:
                result.append(self.item_delim)
            if self.item_header:
                if self.with_index:
                    result.append(f'{self.item_header} {index}) {self.header_delim}')
                else:
                    result.append(f'{self.item_header} {self.header_delim}')
            result.extend(options['fn'](thing))
        result.append(self.block_tail)
        return result


#
# Assume the source have examples, slots, skills, values as well as utterance.
# This prompt template is designed to address template for full skill.
class Prompt:
    def __init__(self, source: str, extra_tokens=[]):
        self.template = Compiler().compile(source)
        self.extra_tokens = extra_tokens
        self.helpers = {
            'list_examples': ObjectLister(item_header="### Examples"),
            'list_skills': ObjectLister(item_header="### Functions", item_delim=",", block_header="[", block_tail="]"),
            'list_slots': ObjectLister(item_header=None, item_delim=",", block_header="[", block_tail="]"),
            'list_values': ObjectLister(item_header=None, item_delim=",", block_header="[", block_tail="]")
        }
        self.partials = {}

    def __call__(self, item: dict[str, any]) -> str:
        # First we need to create the example.
        return self.template(item, helpers=self.helpers, partials=self.partials)


#
# LugPrompts assumes the following prompt template in pybars depending on the following information:
# skills: List[SkillSpec]
# slots: List[SlotSpec]
# exemplars: List[Exemplar]
# values: ?
#
SkillPrompts = {
    "simple_prompt":
        "<s> Convert the input text to structured representation. ### Input: {{utterance}} ### Output:",
    "full_simple_prompt_txt00":
        """
        <s> Given the input sentence, construct a function representation of this sentence, including the function name,
        parameters, and their corresponding values. This function representation should describe the target sentence 
        accurately.  
         
        The function must be one of the following 
        {{#list_skills skills}} {{name}} {{/list_skills}}
        .
        
        For each parameter with its value mentioned in the sentence, enclose the parameter and its corresponding values in
         brackets. The parameters must be one of the following:
        {{#list_slots slots}} {{name}} {{/list_slots}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
    "full_exampled_prompt":
        """
        <s> Given the input sentence, construct a function representation of this sentence, including the function name,
         parameters, and their corresponding values. This function representation should describe the target sentence 
         accurately and the function must be one of the following 
        {{#list_skills skills}} {{name}} {{/list_skills}}
        .
        For each parameter with its value mentioned in the sentence, enclose the parameter and its corresponding values in
         brackets. The parameters must be one of the following:
        {{#list_slots slots}} {{name}} {{/list_slots}}
        The order your list the parameters within the function must follow the order listed above. 
        
        Here are a couple of examples.
        {{#list_examples examples}} Sentence: {{template}} \n Output: {{owner}} \n {{/list_examples}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
    "exampled_prompt_for_skill00":
        """<s> 
        Given an input sentence, a set of functions with names and their descriptions, as well as some example templates
         of how to express these functions in natural language text, the goal is to determine the function implied by 
        the input sentence. The selected function should accurately describe the target sentence, and it should 
        be one of the following functions:

        {{#list_skills skills}} {{owner}} : {{description}} {{/list_skills}} . 
         
        Here are a couple of example templates:
        {{#list_examples examples}} ### Input template: {{template}} \n ### Output: {{owner}} \n {{/list_examples}}
        
        ### Input sentence: \n
        {{utterance}}
        ### Output: \n
        """,

    "exampled_prompt_for_skill01":
        """<s> 
        Given an input sentence, a set of functions with names and their descriptions, as well as some example templates
         of how to express these functions in natural language text, the goal is to determine the function implied by 
        the input sentence. The selected function should accurately describe the target sentence:

        {{#list_skills skills}} {{owner}} : {{description}} {{/list_skills}} . 
         
        Here are a couple of example templates:
        {{#list_examples examples}} ### Input template: {{template}} \n ### Output: {{owner}} \n {{/list_examples}}
        
        ### Input sentence: \n
        {{utterance}}
        ### Output: \n
        """,
}

AllSlotPrompts = {
    "exampled":
        """
        <s> Given the input sentence, specification of a function, including name and description of the function and 
        its parameters, the task is to extract the values for these parameters from the input sentence (if mentioned) 
        and return the extracted parameters and their values in JSON format. \n
       
        Here is the function: \n
        {{#list_skills skills}} {{name}} : {{description}} {{/list_skills}}
        .
        
        For each parameter with its value mentioned in the sentence, extract the mentioned value for these parameters.
        The parameters must be one of the following:
        {{#list_slots slots}} {{name}}: {{description}} {{/list_slots}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
}

# For the slots of enum type, we used different prompt in order to improve the
EnumPrompts = {
        "default" : """"
        <s> Given an input sentence, extract the value for parameter {{name}}, {{description}}, from the input sentence.
        
        Here are possible values for this parameter:
        {{#list_values values}} value {{/list_values}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
}

OneSlotPrompts = {
    "default":
        """
        <s> Given an input sentence, extract the value for parameter {{name}}, {{description}}, from the input sentence.
        
        Here are possible values for this parameter:
        {{#list_values values}} value {{/list_values}}
        
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
    "basic":
        """
        <s> Given an input sentence, extract the value for parameter {{name}}: {{description}} from the input sentence.
        ### Input sentence:
        {{utterance}}
        ### Output:
        """,
}