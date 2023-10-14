#
# Examples assumes that we have potentially more than one example, the goal
# is to create a block for examples.
#
from abc import ABC

from langchain.schema import BaseRetriever
from pybars import Compiler

from core.commons import Prompt, Domain

#
# We will use eos: </s> automatically in both train and decode. Prompt can decide whether
# and how they want to use bos: <s>.
#


# Simple prompt only has utterance.
class SimplePrompt(Prompt, ABC):
    def __init__(self, source):
        self.template = Prompt.compiler.compile(source)

    # Assume the item has ["utterance", "output"], and "utterance" is used to create input.
    def __call__(self, item:dict[str, any]) -> str:
        return self.template(item)


class ObjectLister:
    def __init__(
            self,
            item_header=None,
            header_delim: str = "\n",
            item_delim: str = "\n\n",
            block_header: str = "",
            block_tail: str = "",
            max_size: int = 3,
            with_index: bool = True):
        self.item_header = item_header
        self.max_size = max_size
        self.header_delim = header_delim
        self.item_delim = item_delim
        self.block_header = block_header
        self.block_tail = block_tail
        self.with_index = with_index

    def __call__(self, this, options, items):
        result = []
        result.append(self.block_header)
        for index, thing in enumerate(items[:self.max_size]):
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
#
class ExampledPrompt(Prompt, ABC):
    def __int__(
            self,
            source: str,
            retriever: BaseRetriever,
            domain: Domain,
            topk: int = 3):
        self.source = source
        self.template = Prompt.compiler.compile(source)
        self.retriever = retriever
        self.skills = domain.skills
        self.slots = domain.slots
        self.helpers = {
            'list_examples': ObjectLister(item_header="Example"),
            'list_skills': ObjectLister(item_header=None, item_delim=",", block_header="[", block_tail="]"),
            'list_slots': ObjectLister(item_header="Slot"),
            'list_values': ObjectLister()
        }
        self.partials = {}

    def __call__(self, item: dict[str, any]) -> str:
        # First we need to create the example.
        results = self.retriever.retrieve(item["utterance"])
        item["examples"] = results
        item["skills"] = self.skills
        item["slots"] = self.slots

        return self.template(item, helpers=self.helpers, partials=self.partials)


full_simple_prompt = SimplePrompt("<s> Convert the input text to structured representation. ### Input: {{utterance}} ### Output:")


full_exampled_prompt_full = f"""
    Given a target sentence construct the underlying meaning representation
    of the input sentence as a single function with attributes and attribute
    values. This function should describe the target sentence accurately and the
    function must be one of the following 
    {{#list_skills skills}} {{name}} {{/list_skills}}
    .
    
    The attributes must be one of the following:
    {{#list_slots slots}} {{name}} {{/list_slots}}
    The order your list the attributes within the function must follow the
    order listed above. 
    
    For each attribute, fill in the corresponding value of the attribute 
    within brackets. A couple of examples are below.
    {{#list_examples examples}} Sentence: {{utterance}} \n Output: {{output}} \n {{/list_examples}}
    
    Give the output for the following sentence:
    {{utterance}}
    Output:<s>
    """

if __name__ == "__main__":
    compiler = Compiler()

    # Compile the template
    source = u"{{#list_examples examples}} Sentence: {{utterance}} \n Output: {{output}} \n {{/list_examples}} {{#list_skills skills}} {{name}} {{/list_skills}} "
    template = compiler.compile(source)




    # Add partials
    header = compiler.compile(u'<h1>People</h1>')
    partials = {'header': header}

    # Render the template
    output = template({
        'examples': [
            {'utterance': "Yehuda", 'output': "Katz"},
            {'utterance': "Carl", 'output': "Lerche"}
        ],
        'skills' : [
            {"name": "inform"},
            {"name": "notify"}
        ]
    }, helpers=helpers, partials=partials)

    print(output)

    sys.exit(0)
