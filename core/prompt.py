#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
from pybars import Compiler
import html


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
            item_header_delim: str = "\n",
            item_delim: str = "\n\n",
            block_header: str = "",
            block_tail: str = "",
            with_index: bool = True):
        self.item_header = item_header
        self.item_header_delim = item_header_delim
        self.item_delim = item_delim
        self.block_header = block_header
        self.block_tail = block_tail
        self.with_index = with_index

    def __call__(self, this, options, items):
        result = []
        result.append(self.block_header)
        for index, thing in enumerate(items):
            if self.item_header:
                if self.with_index:
                    result.append(f'{self.item_header} {index}) {self.item_header_delim}')
                else:
                    result.append(f'{self.item_header} {self.item_header_delim}')
            result.extend(options['fn'](thing))
            result.append(self.item_delim)
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
            'list_examples': ObjectLister(block_header="\nThe expression templates are:\n", item_header="Expression template"),
            'list_skills': ObjectLister(block_header="\nThe functions are:\n", item_delim="\n"),
            'list_slots': ObjectLister(item_header=None, item_delim=",", block_header="[", block_tail="]"),
            'list_values': ObjectLister(item_header=None, item_delim=",", block_header="[", block_tail="]")
        }
        self.partials = {}

    def __call__(self, item: dict[str, any]) -> str:
        # First we need to create the example.
        return html.unescape(self.template(item, helpers=self.helpers, partials=self.partials))


#
# LugPrompts assumes the following prompt template in pybars depending on the following information:
# skills: List[SkillSpec]
# slots: List[SlotSpec]
# exemplars: List[Exemplar]
# values: ?
#

SkillPrompts = {
    "specs_exampled":
        Prompt("""Given a set of functions defined by their names, descriptions, and example templates for expressing them in natural language text, determine the function implied by the input sentence.
        
        {{#list_skills skills}} {{name}} : {{description}} {{/list_skills}}
        
        {{#list_examples examples}} ### Input template: {{template}} \n ### Output: {{owner}} </s> \n {{/list_examples}}

### Input sentence: {{utterance}}
### Output:"""),

    "specs_only":
        Prompt("""Given a set of functions defined by their names, descriptions, and example templates for expressing them in natural language text, determine the function implied by the input sentence.
        
        {{#list_skills skills}} {{name}} : {{description}} {{/list_skills}}
        
### Input sentence: {{utterance}} 
### Output:"""),
}

# For the slots of enum type, we used different prompt in order to improve the
EnumPrompts = {
    "default":
        Prompt("""
        Given an input sentence, extract the value for parameter {{name}}, {{description}}, from the input sentence.
        
        Here are possible values for this parameter:
        {{#list_values values}} value {{/list_values}}
        
### Input sentence:
{{utterance}}
### Output:"""),
}

SlotPrompts = {
    "default":
        Prompt("""From an given input sentence, extract the value for parameter {{name}}: {{description}}.
        
        Here are possible values for this parameter:
        {{#list_values values}} value {{/list_values}}
        
### Input sentence: {{utterance}}
### Output:"""),
    "basic":
        Prompt("""From an given input sentence, extract the value for parameter {{name}}: {{description}}.
        
### Input sentence: {{utterance}}
### Output:"""),
}
