import re
import string
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def evaluate_qa(gt, pred):
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)
    em = (pred == gt)
    f1 = f1_score(pred, gt)[0]
    return {'reward': em, 'em': em, 'f1': f1, 'gt': gt, 'pred': pred}


def normalize_strategyqa_answer(s):
    if s.lower() == 'yes':
        return True
    elif s.lower() == 'no':
        return False
    else:
        return s


def evaluate_strategyqa(gt, pred):
    pred_bool = normalize_strategyqa_answer(pred)
    acc = (pred_bool == gt)
    return {'reward': acc, 'acc': acc, 'gt': gt, 'pred': pred}
