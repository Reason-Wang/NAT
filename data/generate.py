import json
import argparse
import logging
import os
import openai
from tqdm import tqdm
from api.openai.completion import ChatModel
from api.openai.utils import get_api_keys
from api.tools import evaluate
import copy
from model.causallm import ChatCausalLM

from data.utils import extract_thought_action, extract_action_args, get_observation, format_question_with_prompt

logging.getLogger().setLevel(logging.ERROR)
from datetime import datetime

task2file = {
    "test":{
        'gsm8k':"test.json",
        'ASDiv':"test_one_answer.jsonl",
        'MultiArith':"test.jsonl",
        'SVAMP':"test.jsonl",
        'hotpotqa':"dev_random_500.json",
        'StrategyQA':"test_random_500.jsonl"
    },
    "train": {
        'gsm8k':"train.json",
        'ASDiv':"",
        'MultiArith':"",
        'SVAMP':"",
        'hotpotqa':"train_random_8000.json",
        'StrategyQA':"train.jsonl"
    }
}


def initialize_info(task_name, gt):
    info = {
        'reward': False,
        'gt': gt
    }
    return info


def get_node_value(task_name, messages, model, gt):
    reflection = ""
    info = initialize_info(task_name, gt)
    # do not use stop token for now
    thought_action = model.generate(messages=messages, stop=None, print_prompt=False)
    # print("Thought_Action: ", thought_action)
    # print("Thought action pair:", thought_action)
    thought, action, error = extract_thought_action(thought_action, reflection)
    # print(f"Thought: {thought}")
    # print(f"Action: {action}")
    # print(f"Error: {error}")
    if error:
        return [], info, True, False


    action_type, action_args, error = extract_action_args(action)
    # print("Action type:", action_type)
    # print("Action args:", action_args)
    # print("Error:", error)
    if error:
        return [], info, True, False

    finished = "finish" in action_type.lower()
    if finished:
        info = evaluate(task_name, gt, action_args)

    # node_reward = info['reward'] if info else False
    obs, error = get_observation(action_type, action_args)
    # print(f"Observation: {obs}")
    # print(f"Error: {error}")

    if error:
        return [], info, True, False

    new_messages = [
        {"role": "assistant", "content": thought_action},
        {"role": "user", "content": f"Observation: {obs}"}
    ]

    return new_messages, info, False, finished


def chain(task_name, messages, model, gt, depth=1, max_depth=6):
    print(f"Depth: {depth}")
    # print(f"Messages: {messages}")

    # value includes chatgpt format prompt
    new_message, info, error, finished = get_node_value(task_name, messages, model, gt)
    print(f"New message: {new_message}")
    # print(f"Info: {info}")
    # print(f"Error: {error}")
    # print(f"Finished: {finished}")
    node = {"conversations": new_message, "info": info}

    if error:
        return node

    if depth >= max_depth:
        print(f"Max depth reached: {depth}")
        return node

    if finished:
        return node
    else: # unfinished
        # print(f"New message: {new_message}")

        new_messages = copy.deepcopy(messages)
        new_messages.extend(new_message)
        children_nodes = chain(task_name, new_messages, model, gt, depth+1, max_depth)
        nodes = copy.deepcopy(node)
        nodes['conversations'].extend(children_nodes['conversations'])
        if 'info' in children_nodes:
            nodes['info'] = children_nodes['info']
        return nodes



def run(
    task_name,
    data_lsit,
    model,
    cache_file=None,
    template=None,
):
    # messages = get_conversation_messages(task_name, shot='one-shot')
    # # questions_messages_list, ground_truths = get_questions_messages_ground_truths(data_lsit, messages)
    if task_name in {'gsm8k', 'ASDiv', 'MultiArith', 'SVAMP'}:
        prompts_path = f"prompts/gsm8k/gsm8k_conversation.json"
    elif task_name in {'hotpotqa'}:
        prompts_path = f"prompts/hotpotqa/hotpotqa_conversation.json"
    elif task_name in {'StrategyQA'}:
        prompts_path = f"prompts/StrategyQA/StrategyQA_conversation.json"
    else:
        raise NotImplementedError(f"Task {task_name} not implemented")
    with open(prompts_path, "r") as fin:
        prompts = json.load(fin)

    questions = [e['question'] for e in data_lsit]
    ground_truths = [e['answer'] for e in data_lsit]
    ids = [e['id'] for e in data_lsit]
    question_messages = [format_question_with_prompt(task_name, question, prompts, template=template) for question in questions]

    print(f"Input messages: {question_messages[0]}")

    trajs = []
    for id, question_message, gt in tqdm(zip(ids, question_messages, ground_truths), total=len(questions)):
        traj = chain(task_name, question_message, model, gt, depth=0)
        traj['id'] = id
        trajs.append(traj)
        if cache_file is not None:
            with open(cache_file, 'a') as f:
                f.write(json.dumps(traj) + '\n')

    return trajs


def evaluate_trajs(task_name, trajs):
    if task_name in {'hotpotqa', 'StrategyQA'}:
        correct_count = 0
        all_f1 = 0.0
        for traj in trajs:
            if traj['info']['reward']:
                correct_count += 1
            if 'f1' in traj['info']:
                all_f1 += traj['info']['f1']
            else:
                all_f1 += 0.0
        return {"em": correct_count / len(trajs), "f1": all_f1 / len(trajs)}
    elif task_name in {'gsm8k', 'ASDiv', 'MultiArith', 'SVAMP'}:
        correct_count = 0
        for traj in trajs:
            if traj['info']['reward']:
                correct_count += 1

        return {'accuracy': correct_count / len(trajs)}
    else:
        raise NotImplementedError(f"Task {task_name} not implemented")


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--task_name', type=str, default='hotpotqa', choices=['gsm8k', 'ASDiv', 'MultiArith', 'SVAMP', 'hotpotqa', 'StrategyQA'])
    args.add_argument('--task_split', type=str, default='train', choices=['train','test'])
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=None)
    args.add_argument('--template', type=str, default='zero-shot')
    args.add_argument('--promptpath', type=str, default='')
    args.add_argument('--suffix', type=str, default='')

    args = args.parse_args()
    return args


# We current only support chatgpt format
if __name__ == '__main__':
    args = parse_args()
    print(args)

    model_name = args.model
    base_model_name = model_name.split('/')[-1]
    outfilename = f"data/trajs/{args.task_name}/{args.task_split}_{args.task_start_index}_{args.task_end_index}_{base_model_name}_{args.temperature}"+args.suffix
    resultfilename = f"data/results/{args.task_name}/{args.task_split}_{args.task_start_index}_{args.task_end_index}_{base_model_name}_{args.temperature}"+args.suffix
    dir_name = os.path.dirname(outfilename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(outfilename)

    data_path = f"data/dataset/{args.task_name}/{task2file[args.task_split][args.task_name]}"

    with open(data_path, "r") as fin:
        if data_path.endswith('.json'):
            data = json.load(fin)
        elif data_path.endswith('.jsonl'):
            data = [json.loads(line) for line in fin]
        else:
            raise NotImplementedError(f"Data format {data_path} not implemented")

        if 'id' in data[0]:
            data_indexes = [e['id'] for e in data][args.task_start_index:args.task_end_index]
            data_list = data[args.task_start_index:args.task_end_index]
        else:
            data_indexes = list(range(len(data)))[args.task_start_index:args.task_end_index]
            data_list = data[args.task_start_index:args.task_end_index]

    if 'gpt' in model_name:
        openai.api_key = get_api_keys("./data/keys.json")[0]
        model = ChatModel(model=model_name, temperature=args.temperature, stop=["\nObservation: "])
        assert args.template in ['one-shot','one-shot2']
    else:
        model = ChatCausalLM(
            model_name,
            max_new_tokens=512,
            temperature=args.temperature,
            device="auto",
            cache_dir=None,
            conversation_template="llama-2",
        )

    print("Model: ", model)

    trajs = run(
        task_name=args.task_name,
        data_lsit=data_list,
        model=model,
        cache_file=outfilename+"_cache.json",
        template=args.template
    )

    with open(outfilename+".json", "w") as fout:
        json.dump(trajs, fout, indent=2)

    results = evaluate_trajs(args.task_name, trajs)
    print(results)
    dir_name = os.path.dirname(resultfilename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(resultfilename+".json", "w") as fout:
        json.dump(results, fout, indent=2)

    # To aggregrate all results in one file
    if args.task_split == "test":
        results['task'] = args.task_name
        with open(os.path.join(model_name, "evaluate_results.jsonl"), "a") as fout:
            fout.write(json.dumps(results)+"\n")