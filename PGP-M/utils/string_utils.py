import torch

class mutation_SuffixManager:
    def __init__(self, *, tokenizer, instruction, target, adv_string):

        self.tokenizer = tokenizer
        # self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())

        messages = [
            {"role": "user", "content": self.adv_string},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks)

        return input_ids