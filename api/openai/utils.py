import json
from typing import List


def get_api_keys(key_path_or_keys):
    # A json file containing list of keys
    api_keys = []
    if isinstance(key_path_or_keys, str):
        with open(key_path_or_keys, 'r') as f:
            api_keys = json.load(f)['openai_api_keys']
    elif isinstance(key_path_or_keys, List):
        api_keys = key_path_or_keys
    else:
        raise TypeError("Must be a path of a json file containing keys or list of keys.")

    return api_keys
