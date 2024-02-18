import re


def extract_answer(completion):
    match = re.search(r"#### (\-?[0-9\.\,]+)", completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return completion


def is_correct(prediction, answer):
    gt_answer = extract_answer(answer)
    assert gt_answer is not None
    return prediction == gt_answer


# def evaluate_gsm8k(gt, pred):
#     # pred = extract_answer(pred)
#     gt = extract_answer(gt)
#     correct = (pred == gt)
#     return {'reward': correct, 'gt': gt, 'pred': pred}


def evaluate_math(gt, pred):
    gt = extract_answer(gt)
    correct = (pred == gt)
    return {'reward': correct, 'gt': gt, 'pred': pred}