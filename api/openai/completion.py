import time
from typing import List
import openai
from tqdm import tqdm


def request(
        model: str,
        messages: List,
        num_retries: int = 5,
        return_dict: bool = True,
        **kwargs
):
    response = {}
    # retry request (handles connection errors, timeouts, and overloaded API)
    for i in range(num_retries):
        try:
            # print(prompt)
            # print(max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty)
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                **kwargs
            )
            response['success'] = True
            break
        except Exception as e:
            response['success'] = False
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            time.sleep(10)
    # TODO: logic bug, case: response['success'] = False
    if return_dict:
        return response
    else:
        return response['choices'][0]['message']['content']


class ChatModel:
    def __init__(self, model: str, temperature: float, stop: List[str]=None):
        self.model = model
        self.temperature = temperature
        self.stop = stop

    def generate(self, messages: List, temperature: float = None, stop: List[str] = None, print_prompt=False):
        if print_prompt:
            print(messages)
        response = request(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            return_dict=False,
            stop=self.stop if stop is None else stop
        )
        return response

