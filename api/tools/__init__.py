# from .search import search
from api.tools.math import calculate
from api.tools.math.utils import evaluate_math
from api.tools.qa.utils import evaluate_qa, evaluate_strategyqa
from api.tools.search import GOOGLESearch
from api.tools.search.google_search import google_search_serper_with_answer, GOOGLESearchTitlesSnippets


def evaluate(task_name, gt, prediction):
    if task_name in {"gsm8k", "ASDiv", "MultiArith", "SVAMP"}:
        return evaluate_math(gt, prediction)
    elif task_name == "hotpotqa":
        return evaluate_qa(gt, prediction)
    elif task_name == "StrategyQA":
        return evaluate_strategyqa(gt, prediction)
    else:
        raise ValueError("Unknown task name: {}".format(task_name))


def call_tools(tool_name, tool_input):
    print(f"tool_name: {tool_name}, tool_input: {tool_input}")
    result = None
    if tool_name == "search":
        result = GOOGLESearchTitlesSnippets(tool_input)
        # result = google_search_serper_with_answer(tool_input)
        # print("Context: ", result)
    elif tool_name == "calculate":
        result = calculate(tool_input)
    else:
        raise ValueError("Unknown tool name: {}".format(tool_name))

    return result
