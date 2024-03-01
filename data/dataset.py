import warnings
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from prompts.conversations import get_conv_template


def preprocess(organized_data, tokenizer, conv_template_name, max_length=4096):
    human_role_set = {"human", "user"}
    ai_role_set = {"ai", "gpt", "assistant"}
    all_input_ids = []
    all_targets = []
    all_attention_masks = []

    # Get the role label and content label
    if 'from' in organized_data[0]['conversations'][0]:
        role_label, content_label = "from", "value"
    elif 'role' in organized_data[0]['conversations'][0]:
        role_label, content_label = "role", "content"
    else:
        raise ValueError("Cannot find role label and content label in the data.")

    for i, data in enumerate(organized_data):
        # print(len(data))
        conv = get_conv_template(conv_template_name)
        conversation = data['conversations']
        for j in range(len(conversation)):
            conv.append_message(conv.roles[0] if conversation[j][role_label] in human_role_set
                                else conv.roles[1], conversation[j][content_label], conversation[j]['loss'])

        separate_prompts = conv.get_separate_prompt_with_to_loss()

        # all_prompt = ""
        input_ids = []
        targets = []
        for i, (prompt, to_loss) in enumerate(separate_prompts):
            # print(prompt)
            # all_prompt += prompt
            if i == 0:
                prompt = tokenizer.bos_token + prompt

            tmp_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if to_loss:
                tmp_target = tmp_input_ids.copy()
            else:
                tmp_target = [-100] * len(tmp_input_ids)
            input_ids.extend(tmp_input_ids)
            targets.extend(tmp_target)

        input_ids = input_ids[:max_length]
        targets = targets[:max_length]
        if input_ids[-1] != tokenizer.eos_token_id and len(input_ids) < max_length:
            input_ids.append(tokenizer.eos_token_id)
            # TODO: check if this is correct
            targets.append(tokenizer.eos_token_id)

        all_input_ids.append(input_ids)
        all_targets.append(targets)
        all_attention_masks.append([1] * len(input_ids))

    return dict(
        input_ids=all_input_ids,
        labels=all_targets,
        attention_mask=all_attention_masks
    )


class ConversationDataset(Dataset):
    '''
    :param organized_data: List of conversations. Each conversation is a list of turns.
    :param conv_template_name: Name of the prompt template.
    '''

    def __init__(self, tokenizer, organized_data, conv_template_name, task_name, max_length=4096,  positive_only=True):
        super(ConversationDataset, self).__init__()
        self.tokenizer = tokenizer
        if positive_only:
            positive_data = []
            for data in organized_data:
                if task_name == "gsm8k":
                    if data['info']['reward']:
                        positive_data.append(data)
                #
                elif task_name == "hotpotqa":
                    if data['info']['reward']:
                        positive_data.append(data)
                elif task_name == 'StrategyQA':
                    if data['info']['reward']:
                        positive_data.append(data)
                else:
                    raise ValueError(f"{task_name} is not supported.")
            organized_data = positive_data
            warnings.warn(f"Only positive data is used. Data size: {len(organized_data)}")
        else:
            warnings.warn(f"Both positive and negative data is used. Data size: {len(organized_data)}")
        data_dict = preprocess(organized_data, tokenizer, conv_template_name, max_length)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

        self.has_print = False

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        if not self.has_print:
            print(self.tokenizer.decode(self.input_ids[item]))
            self.has_print = True
        return dict(
            input_ids=torch.tensor(self.input_ids[item]),
            labels=torch.tensor(self.labels[item]),
            attention_mask=torch.tensor(self.attention_mask[item])
        )


class LazyConversationDataset(Dataset):
    def __init__(self, tokenizer, sources, targets, max_length):
        # Not implemented yet
        raise NotImplementedError
        super(LazyConversationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.sources = sources
        self.targets = targets
        self.max_length = max_length
        self.has_print = False

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        pass


class CollatorWithPadding(object):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = [e['input_ids'] for e in instances]
        attention_masks = [e['attention_mask'] for e in instances]
        labels = [e['labels'] for e in instances]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_masks,
            labels=padded_labels
        )

