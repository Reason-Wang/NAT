import copy
import json
import random
from deprecated import deprecated

from api.tools import call_tools


def extract_thought_action(thought_action, reflection):
    error = False
    thought = "Thought: None"
    action = "Action: None"
    try:
        if "\nAction: " in thought_action.strip():
            thought, action = thought_action.strip().split("\nAction: ")[:2]
        elif "Action: " in thought_action.strip():
            thought = ""
            action = thought_action[len("Action: "):]
        else:
            thought = thought_action.split("\n")[0]
            action = None
            # will skip bad ids
            error = True
        if len(reflection) > 0:
            thought = reflection.strip() + " " + thought
    except Exception as e:
        print("Error while trying to extract thought action pair: ", e)
        error = True

    return thought, action, error
#
#
def extract_action_args(action):
    action_type = None
    action_args = None
    error = False
    try:
        action_type, action_args = action.split('[')[:2]
        action_args = action_args[:-1]
    except Exception as e:
        print("Error while trying to extract action and arguments.")
        error = True
    return action_type, action_args, error


def get_observation(action_type, action_args):
    obs = "Observation: None"
    error = False
    if "finish" not in action_type.lower():
        try:
            obs = call_tools(action_type, action_args)
        # We expect this not happen
        except Exception as exc:
            print('%r generated an exception: %s' % ((action_type, action_args), exc))
            error = True
    else:
        assert obs == "Observation: None", f"action {action_type} has observation {obs}"
        obs = f"Episode finished"

    return obs, error


@deprecated(reason="Transfer to load_data_with_prompts")
def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    # data_list = []
    # for k, v in data.items():
    #     data_list.extend(v)
    if isinstance(data, list):
        data_list = data

    for data_dict in data_list:
        if not 'conversations' in data_dict:
            data_dict['conversations'] = data_dict['items']
            del data_dict['items']

    ai_role_set = {"ai", "gpt", "assistant"}
    for data_dict in data_list:
        for conversation in data_dict['conversations']:
            if 'loss' not in conversation:
                if conversation['from'] in ai_role_set:
                    conversation['loss'] = True
                else:
                    conversation['loss'] = False

    return data_list


def load_pos_neg_data_with_prompts(
        task_name,
        pos_path,
        neg_path,
        prompt_path,
        template,
        question_path,
        pos_num,
        neg_num):
    # For positive data, we use the same, for negative data, we randomly sample
    pos_trajs = load_data_with_prompts(task_name, pos_path, prompt_path, template, question_path, filter=False)[:pos_num]
    neg_trajs = load_data_with_prompts(task_name, neg_path, prompt_path, template, question_path, filter=False)
    # set seed
    random.seed(42)
    neg_trajs = random.sample(neg_trajs, neg_num)

    return pos_trajs + neg_trajs


def load_pos_neg_constrastive_data_with_prompts(
        task_name,
        pos_path,
        neg_overlap_path,
        neg_nonoverlap_path,
        prompt_path,
        template,
        question_path,
        pos_num,
        neg_overlap_num,
        neg_nonoverlap_num
):
    # Set filter to False since we train for gsm8k now
    # We have to make sure that all samples are with same index
    # So we could randomly sample indexs and then get the samples
    pos_trajs = load_data_with_prompts(task_name, pos_path, prompt_path, template, question_path, filter=False)[:pos_num]
    neg_overlap_trajs = load_data_with_prompts(task_name, neg_overlap_path, prompt_path, template, question_path, filter=False)[:neg_overlap_num]
    neg_nonoverlap_trajs = load_data_with_prompts(task_name, neg_nonoverlap_path, prompt_path, template, question_path, filter=False)[:neg_nonoverlap_num]

    return pos_trajs + neg_overlap_trajs + neg_nonoverlap_trajs


def load_data_with_prompts(
        task_name,
        data_path,
        prompt_path: str,
        template: str,
        question_path: str,
        filter=False,
):
    '''
    We now use this to load raw data and combine them with prompts
    All data must contain 'conversations' key, which is a list of messages
    And a message must contain 'role', 'content' and 'loss' keys
    :param data_path:
    :param prompt_path:
    :param shot:
    :param question_path:
    :param filter: If True, filter out unfinished samples
    :return:
    '''
    # Currently default value is not supported
    with open(data_path, 'r') as f:
        trajs = json.load(f)

    filtered_trajs = []
    # Filter out unfinished samples
    if filter and task_name == 'hotpotqa':
        for traj in trajs:
            if 'pred' in traj['info']:
                filtered_trajs.append(traj)
    else:
        filtered_trajs = trajs

    with open(prompt_path, 'r') as f:
        prompts = json.load(f)

    if question_path.endswith('.json'):
        with open(question_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    elif question_path.endswith('.jsonl'):
        with open(question_path, 'r', encoding='utf-8') as f:
            questions_data = [json.loads(line) for line in f]

    questions_dict = {}
    for q in questions_data:
        questions_dict[q['id']] = q
        # questions = [e['question'] for e in questions_data]

    formatted_trajs = []
    for traj in filtered_trajs:
        formatted_traj = format_sample_with_prompt(task_name, traj, questions_dict[traj['id']]['question'], prompts, template)
        formatted_trajs.append(formatted_traj)

    return formatted_trajs


def format_sample_with_prompt(task_name, traj, question, prompts, template):
    prompt = prompts[template]
    if template in {'zero-shot', 'cot-zero-shot'}:
        prompt_messages = prompt['conversations']
        question_prompt = prompt['question_prompt']
        for message in prompt_messages:
            message['loss'] = False

        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif template in {
        'zero-shot-target-aware',
        'zero-shot-target-aware-2',
        'zero-shot-target-aware-meaningless',
        'cot-zero-shot-target-aware',
        'zero-shot-target-aware-inverse',
        'zero-shot-target-aware-prefix-correct',
        'zero-shot-target-aware-prefix-good',
        'zero-shot-target-aware-prefix-laptop',
        'zero-shot-target-aware-prefix-AB',
        'zero-shot-target-aware-prefix-correct-inverse',
        'zero-shot-target-aware-good',
        'zero-shot-target-aware-random-sentence',
    }:
        prompt_messages = prompt['conversations']
        if task_name in {'gsm8k', 'ASDiv', 'MultiArith', 'SVAMP'}:
            question_prompt = prompt['correct_question_prompt'] if traj['info']['reward'] else prompt['incorrect_question_prompt']
        elif task_name in {'hotpotqa', "StrategyQA"}:
            question_prompt = prompt['correct_question_prompt'] if traj['info']['reward'] else prompt['incorrect_question_prompt']
        else:
            raise NotImplementedError(f"Task {task_name} is not supported.")

        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif template in {'zero-shot-target-aware-soft'}:
        assert task_name == 'hotpotqa' and 'f1' in traj['info']
        prompt_messages = prompt['conversations']
        if traj['info']['f1'] > 0.2:
            question_prompt = prompt['correct_question_prompt']
        elif traj['info']['f1'] <= 0.2:
            question_prompt = prompt['incorrect_question_prompt']
        else:
            raise ValueError(f"Invalid f1 score: {traj['info']['f1']}")
        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif template in {'zero-shot-target-aware-three-class'}:
        assert task_name == 'hotpotqa' and 'f1' in traj['info']
        prompt_messages = prompt['conversations']
        if traj['info']['f1'] == 0.0:
            question_prompt = prompt['class_1_question_prompt']
        elif traj['info']['f1'] > 0.0 and traj['info']['f1'] < 1.0:
            question_prompt = prompt['class_2_question_prompt']
        elif traj['info']['f1'] == 1.0:
            question_prompt = prompt['class_3_question_prompt']
        else:
            raise ValueError(f"Invalid f1 score: {traj['info']['f1']}")
        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif template in {'zero-shot-target-aware-four-class'}:
        assert task_name == 'hotpotqa' and 'f1' in traj['info']
        prompt_messages = prompt['conversations']
        if traj['info']['f1'] == 0.0:
            question_prompt = prompt['class_1_question_prompt']
        elif traj['info']['f1'] > 0.0 and traj['info']['f1'] < 0.4:
            question_prompt = prompt['class_2_question_prompt']
        elif traj['info']['f1'] >= 0.4 and traj['info']['f1'] < 1.0:
            question_prompt = prompt['class_3_question_prompt']
        elif traj['info']['f1'] == 1.0:
            question_prompt = prompt['class_4_question_prompt']
        else:
            raise ValueError(f"Invalid f1 score: {traj['info']['f1']}")
        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    else:
        raise NotImplementedError(f"Prompt {template} is not supported")

    formatted_traj = copy.deepcopy(traj)
    formatted_traj['conversations'] = prompt_messages_with_question + formatted_traj['conversations']

    return formatted_traj


def format_question_with_prompt(task_name, question, prompts, template):
    prompt = prompts[template]
    if template == 'zero-shot' or template == 'cot-zero-shot':
        prompt_messages = prompt['conversations']
        question_prompt = prompt['question_prompt']
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
            }
        ]
    elif template in {
        'zero-shot-target-aware',
        'zero-shot-target-aware-2',
        'zero-shot-target-aware-meaningless',
        'cot-zero-shot-target-aware',
        'zero-shot-target-aware-prefix-correct',
        'zero-shot-target-aware-prefix-good',
        'zero-shot-target-aware-prefix-laptop',
        'zero-shot-target-aware-prefix-AB',
        'zero-shot-target-aware-prefix-correct-inverse',
        'zero-shot-target-aware-good',
        'zero-shot-target-aware-random-sentence',
        'zero-shot-target-aware-inverse',
    }:
        prompt_messages = prompt['conversations']
        # This is for inference where we let model to generate correctly
        question_prompt = prompt['correct_question_prompt']
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
            }
        ]
    elif template in ['one-shot','one-shot2']:
        prompt_messages = prompt['conversations']
        question_prompt = prompt['question_prompt']
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
            }
        ]
    elif template in {'zero-shot-target-aware-soft'}:
        assert task_name == 'hotpotqa'
        prompt_messages = prompt['conversations']
        question_prompt = prompt['correct_question_prompt']
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif template in {'zero-shot-target-aware-three-class'}:
        assert task_name == 'hotpotqa'
        prompt_messages = prompt['conversations']
        question_prompt = prompt['class_3_question_prompt']
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
            }
        ]
    elif template in {'zero-shot-target-aware-four-class'}:
        assert task_name == 'hotpotqa'
        prompt_messages = prompt['conversations']
        question_prompt = prompt['class_4_question_prompt']
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
            }
        ]
    else:
        raise NotImplementedError(f"Prompt {template} is not supported")

    return prompt_messages_with_question


