import json

import openai
from tqdm import tqdm
from typing import List
import api.openai.completion


def generate_with_prompts(
        api_keys: List[str],
        task: str,
        model: str,
        messages_list: List,
        start_idx=0,
        end_idx=None,
        cache_file: str=None,
        **kwargs
):
    responses = []
    # TODO: Add multiprocess
    for i, messages in enumerate(tqdm(messages_list)):
        if i < start_idx:
            continue
        if end_idx is not None and i >= end_idx:
            break

        openai.api_key = api_keys[i % len(api_keys)]
        response = api.openai.completion.request(model=model, messages=messages, **kwargs)
        response['generation_index'] = i
        responses.append(response)
        if cache_file is not None:
            with open(cache_file, 'a') as f:
                f.write(json.dumps(response)+'\n')
    return responses

