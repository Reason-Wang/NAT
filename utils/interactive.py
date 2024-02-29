import gc
import json
from typing import Dict, Iterable

import click
import huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from termcolor import colored
from data.utils import extract_thought_action, extract_action_args, get_observation


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


def is_sentence_complete(output: str):
    """Check whether the output is a complete sentence."""
    end_symbols = (".", "?", "!", "...", "。", "？", "！", "…", '"', "'", "”")
    return output.endswith(end_symbols)


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    if hasattr(model, "device"):
        device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    logprobs = params.get("logprobs", None)  # FIXME: Support logprobs>1.
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids


    max_src_len = context_len - max_new_tokens - 1


    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)


    start_ids = torch.as_tensor([input_ids], device=device)

    past_key_values = out = None
    token_logprobs = [None]  # The first token has no logprobs.
    sent_interrupt = False
    finish_reason = None
    stopped = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill

            out = model(input_ids=start_ids, use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])
        else:  # decoding

            out = model(
                input_ids=torch.as_tensor(
                    [[token] if not sent_interrupt else output_ids],
                    device=device,
                ),
                use_cache=True,
                past_key_values=past_key_values if not sent_interrupt else None,
            )
            sent_interrupt = False
            logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]


        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)
        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs
                    if echo
                    else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}]
                    * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "logprobs": ret_logprobs,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    else:
        finish_reason = "length"

    if stopped:
        finish_reason = "stop"

    yield {
        "text": output,
        "logprobs": ret_logprobs,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()


def obtain_observation(response):
    reflection = ""
    thought_action = response
    thought, action, error = extract_thought_action(thought_action, reflection)
    if error:
        return "", True, False

    action_type, action_args, error = extract_action_args(action)
    if error:
        return "", True, False

    finished = "finish" in action_type.lower()
    obs, error = get_observation(action_type, action_args)

    if error:
        return "", True, False

    return obs, False, finished


@click.command()
@click.option('--model_name_or_path', type=str, help='Model name or path')
@click.option('--task', type=str, help='Task name')
def main(
    model_name_or_path: str,
    task: str
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    if task == "math":
        with open("prompts/gsm8k/gsm8k_conversation.json", "r") as f:
            prompt = json.load(f)['zero-shot-target-aware']
            question_prompt = prompt['correct_question_prompt']
    elif task in {"hotpotqa", "qa"}:
        with open("prompts/hotpotqa/hotpotqa_conversation.json", "r") as f:
            prompt = json.load(f)['zero-shot-target-aware-four-class']
            question_prompt = prompt['class_4_question_prompt']
    else:
        raise NotImplementedError(f"Task {task} not implemented")

    from prompts.conversations import get_conv_template
    conv = get_conv_template("llama-2")
    for turn in prompt['conversations']:
        conv.append_message(conv.roles[0] if turn['role'] == 'user' else conv.roles[1], turn['content'])
    question = input("Input question: ")
    question = question_prompt.format(question=question)
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    max_turn = 8
    current_turn = 0
    while True:
        current_turn += 1
        if current_turn >= max_turn:
            break
        prompt = conv.get_prompt()
        params = {
            "prompt": prompt
        }
        prev = len(prompt)
        output = ""
        print(colored("NAT: ", "green"), end="", flush=True)
        for i, result in enumerate(generate_stream(model, tokenizer, params, "cuda", 1024)):
            output = result['text']
            print(output[prev:], end="", flush=True)
            prev = len(result["text"])
        response = output[len(prompt):]
        conv.update_last_message(response)
        obs, error, finished = obtain_observation(response)
        obs = f"Observation: {obs}"
        if error:
            break
        if finished:
            break

        print("\n"+colored("Environment: ", "green")+obs+"\n", end="", flush=True)
        conv.append_message(conv.roles[0], obs)
        conv.append_message(conv.roles[1], None)
    print("\n")


if __name__=="__main__":
    main()

