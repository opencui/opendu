#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
import html

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
        item_header_delim: str = "\n",
        item_delim: str = "\n",
        block_header: str = "",
        block_tail: str = "",
        with_index: bool = True,
    ):
        self.item_header = item_header
        self.item_header_delim = item_header_delim
        self.item_delim = item_delim
        self.block_header = block_header
        self.block_tail = block_tail
        self.with_index = with_index

    def __call__(self, this, options, items):
        result = []
        # If the item list is empty.
        if len(items) == 0:
            return result
        result.append(self.block_header)
        for index, thing in enumerate(items):
            if self.item_header:
                if self.with_index:
                    result.append(
                        f"{self.item_header} {index}) {self.item_header_delim}"
                    )
                else:
                    result.append(f"{self.item_header} {self.item_header_delim}")
            result.extend(options["fn"](thing))
            result.append(self.item_delim)
        result.append(self.block_tail)
        return result


#
# Assume the source have examples, slots, skills, values as well as utterance.
# This prompt template is designed to address template for full skill.
class Prompt:
    def __init__(self, source: str):
        self.template = Compiler().compile(source)
        self.extra_tokens = []
        self.helpers = {
            "list_examples": ObjectLister(),
            "list_skills": ObjectLister(
                block_header="\nGiven the definition for the following functions:\n"
            ),
            "list_skill_names": ObjectLister(
                block_header="Classify the input into the following functions:\n"
            ),
            "list_slots": ObjectLister(
                item_header=None, item_delim=",", block_header="[", block_tail="]"
            ),
            "list_values": ObjectLister(
                item_header=None,
                item_delim=",",
                block_header="Given the candidate values: [",
                block_tail="null]",
            ),
        }
        self.partials = {}

    def __call__(self, item: dict[str, any]) -> str:
        # First we need to create the example.
        return html.unescape(
            self.template(item, helpers=self.helpers, partials=self.partials)
        )


#
# LugPrompts assumes the following prompt template in pybars depending on the following information:
# skills: List[SkillSpec]
# slots: List[SlotSpec]
# exemplars: List[Exemplar]
# values: ?
#

MulticlassSkillPrompts = {
    "specs_only": Prompt(
        "{{#list_skills skills}} {{name}}: {{description}} {{/list_skills}}\n"
        'Classify the input sentence: "{{utterance}}" as one of '
        '{{#list_skill_names skills}} "{{name}}" {{/list_skill_names}}.\n'
        "The prediction is:"
    ),
    "specs_exampled": Prompt(
        '{{#list_skills skills}} "{{name}}": {{description}} {{/list_skills}}\n'
        "{{#list_examples examples}}Input template: {{template}} means {{owner}}\n{{/list_examples}}\n"
        'Classify the input sentence: "{{utterance}}" as one of '
        '{{#list_skill_names skills}} "{{name}}" {{/list_skill_names}}.\n'
        "The prediction is:"
    ),
    "full": Prompt(
        '{{#list_skills skills}} "{{name}}": {{description}} {{/list_skills}}\n'
        "Classify the input as one of "
        '{{#list_skill_names skills}} "{{name}}" {{/list_skill_names}}.\n\n'
        '{{#list_examples examples}}Input: "{{template}}"\nOutput:"{{owner}}"\n{{/list_examples}}\n'
        'Input: "{{utterance}}"\nOutput:'
    ),
    "basic": Prompt(
        '{{#list_skill_names skills}} "{{name}}": {{description}} {{/list_skill_names}}\n'
        "Output null if input does not imply any one of them.\n\n"
        '{{#list_examples examples}}Input: "{{template}}"\nOutput:"{{owner}}"\n{{/list_examples}}\n'
        'Input: "{{utterance}}"\nOutput:'
    ),
}

BinarySkillPrompts = {
    "default": Prompt(
        'Determine whether the input means "{{skill.name}}": {{skill.description}}, output true or false.\n'
        '{{#list_examples examples}}Input: [{{template}}] means "{{target}}"? Output: {{decision}}{{/list_examples}}'
        'Input: [{{utterance}}] means "{{skill.name}}"? Output: '
    ),
    "natural": Prompt(
        'Determine whether the input means "{{skill.name}}": {{skill.description}}, output true or false.\n'
        "{{#list_examples examples}}"
        'Question: Does "{{template}}" mean "{{target}}"? Answer: {{decision}}'
        "{{/list_examples}}"
        'Question: Does "{{utterance}}" mean "{{skill.name}}"? Answer:'
    ),
}


DescriptionPrompts = {
    "default": Prompt(
        'Is it true that "{{utterance}}" fits the description "{{skill.description}}"?'
    ),
    "structural": Prompt(
        '### Instruction:\n'
        'Decide whether the input fit the function {{skill.name}} with description: {{skill.description}}.'
        '### Input:\n'
        '{{utterance}}'
        '### Answer(only in "true" or "false"):'
    ),
    "struct-short": Prompt(
        'Given:\n'
        'the utterance: {{utterance}} \n'
        'the function description: {{skill.description}}\n'
        'Is it true that the utterance fits the function description?<s>'
    ),
    "token": Prompt(
        '<utterance> {{utterance}} </utterance> <func> {{skill.name}} : {{skill.description}} </func> <description>'
    ),
    "struct-token": Prompt(
        'Given the function '
        '<func_name> {{skill.name}} </func_name> with its description: <func_desc> {{skill.description}} </func_desc>, '
        'decide whether <utterance> {{utterance}} </utterance> means this function.'
        'The answer is'
    ),
    "struct-token1": Prompt(
        'Given:\n'
        'the utterance: <utterance> {{utterance}} </utterance>\n'  
        'the function description: <func_desc> {{skill.description}} </func_desc>\n'
        'Is it true that the utterance fit the function description?'
        '<desc> The answer is '
    ),
    "struct-token2": Prompt(
        'Given:\n'
        'the utterance: <utterance> {{utterance}} </utterance>\n'
        'the function description: <func_desc> {{skill.description}} </func_desc>\n'
        'Is it true that the utterance fit the function description?'
        '<desc> '
    ),
}

# This should have the same exact key as the above prompt dictionary.
ExemplarPrompts = {
    "default": Prompt(
        'Is it true that "{{utterance}}" means the same as the example: "{{template}}"?'
    ),
    "structural": Prompt(
        '### Instruction:\n'
        'Decide whether the input means the same as the template: {{template}}.'
        '### Input:\n'
        '{{utterance}}'
        '### Answer(only in "true" or "false"):'
    ),
    "struct-short": Prompt(
        'Given:\n'
        'the template: {{template}}. \n'
        'the input: {{utterance}} \n'
        'Is it true that the utterance means the same as the template?<s>'
    ),

    "token": Prompt(
        '<utterance> {{utterance}} </utterance> <exemplar> {{template}} </exemplar> <exemplar>'
    ),
    "struct-token": Prompt(
        'Given the template <template> {{template}} </template>, '
        'decide whether <utterance> {{utterance}} </utterance> means the same as this template.'
        'The answer is '
    ),
    "struct-token1": Prompt(
        'Given:\n'
        'the utterance: <utterance> {{utterance}} </utterance>\n'  
        'the template: <template> {{template}} </template>\n'
        'Is it true that the utterance means the same as the template?'
        '<exemplar> The answer is '
    ),
    "struct-token2": Prompt(
        'Given:\n'
        'the utterance: <utterance> {{utterance}} </utterance>\n'
        'the template: <template> {{template}} </template>\n'
        'Is it true that the utterance means the same as the template?'
        '<exemplar>'
    ),
}

# For the slots of enum type, we used different prompt in order to improve the
ExtractiveSlotPrompts = {
    "default": Prompt(
        "{{#list_values values}} {{value}} {{/list_values}}\n"
        'The value for {{name}} ({{description}}) from "{{utterance}}" is:'
    ),
    "basic": Prompt(
        'The value for {{name}} ({{description}}) from "{{utterance}}" is:'
    ),
}

NliPrompts = {
    "default": Prompt(
        'Given the premise: "{{premise}}", the hypothesis: "{{hypothesis}}" is its '
    ),
    "boolq": Prompt(
        'Given the premise: "{{premise}}", the hypothesis: "{{hypothesis}}" is its '
    ),
}

BoolPrompts = {
    "default": Prompt('{{label}}</s>'),
    "plain": Prompt('{{label}}'),
    "truefalse": Prompt('{{label}}</true_false_response></s>'),
    "yesno": Prompt('{{label}}</yes_no_response></s>')
}
