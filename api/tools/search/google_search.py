import json
import os

import requests
from api.tools.search.utils import scrape_text, filter_text, split_into_paragraphs, retrieve, rerank
import fcntl
import timeout_decorator



SEARCH_DATABASE = None
HIT = 0
CALL_SERPER = 0

with open("data/keys.json", "r") as f:
    keys = json.load(f)
    GOOGLE_API_KEY = keys["google_api_key"]
    CUSTOM_SEARCH_ENGINE_ID = keys["custom_search_engine_id"]
    SERPER_API_KEY = keys["serper_api_key"]


@timeout_decorator.timeout(5, timeout_exception=TimeoutError)
def req(query, n=10):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": n,
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    result = requests.request("POST", url, headers=headers, data=payload).json()
    return result


def GOOGLESearchTitlesSnippets(query):
    global SEARCH_DATABASE
    global HIT
    global CALL_SERPER
    CALL_SERPER += 1
    if SEARCH_DATABASE is None:
        SEARCH_DATABASE = {}
        # check if file exists
        if not os.path.exists("data/search_database_old.jsonl"):
            open("data/search_database_old.jsonl", "w").close()
        # lock file
        with open("data/search_database_old.jsonl", "r") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            database = [json.loads(line) for line in f]
            for d in database:
                SEARCH_DATABASE[d['key']] = d['value']
            fcntl.flock(f, fcntl.LOCK_UN)

    if query in SEARCH_DATABASE:
        HIT += 1
        print(f"Hit: {HIT} Call: {CALL_SERPER} Hit Rate: {HIT/CALL_SERPER}")
        return SEARCH_DATABASE[query]
    else:
        print(f"Hit: {HIT} Call: {CALL_SERPER} Hit Rate: {HIT / CALL_SERPER}")

        for i in range(10):
            try:
                raw_results = req(query, 10)
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        results = []
        for result in raw_results['organic']:
            results.append({
                "title": result['title'],
                "snippet": result['snippet'] if 'snippet' in result else "",
                "url": result['link']
            })

        titles = [r["title"] for r in results]
        snippets = [r["snippet"] for r in results]
        # relevant_titles, relevant_snippets, _ = rerank(query, titles, snippets, 5)
        # relevant_paragraphs, _ = retrieve(query, snippets, 5)
        _, idxs = retrieve(query, snippets, 5)
        relevant_titles = [titles[i] for i in idxs]
        relevant_snippets = [snippets[i] for i in idxs]
        relevant_titles, relevant_snippets, _ = rerank(query, relevant_titles, relevant_snippets, 3)
        # Will add title be better?
        context = ""
        for i, p in enumerate(relevant_snippets):
            context += f" #{str(i+1)}: " + p

        SEARCH_DATABASE[query] = context
        with open("data/search_database_old.jsonl", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps({"key": query, "value": context}) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

        return context


def google_search_serper_with_answer(query, n=10):
    global SEARCH_DATABASE
    global HIT
    global CALL_SERPER
    CALL_SERPER += 1
    if SEARCH_DATABASE is None:
        SEARCH_DATABASE = {}
        # check if file exists
        if not os.path.exists("data/search_database_serper_new.jsonl"):
            open("data/search_database_serper_new.jsonl", "w").close()
        # lock file
        with open("data/search_database_serper_new.jsonl", "r") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            database = [json.loads(line) for line in f]
            for d in database:
                SEARCH_DATABASE[d['key']] = d['value']
            fcntl.flock(f, fcntl.LOCK_UN)


    if query in SEARCH_DATABASE:
        HIT += 1
        print(f"Hit: {HIT} Call: {CALL_SERPER} Hit Rate: {HIT/CALL_SERPER}")
        return SEARCH_DATABASE[query]
    else:
        print(f"Hit: {HIT} Call: {CALL_SERPER} Hit Rate: {HIT / CALL_SERPER}")

        for i in range(10):
            try:
                result = req(query, n)
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

        if "answerBox" in result:
            if "answer" in result["answerBox"]:
                context = f"[Google Answer]: {result['answerBox']['answer']}"
            else:
                context = f"[Google Snippet]: {result['answerBox']['snippet']}"
        else:
            results = result.get("organic", [])[:3] # Choose top 5 result
            snippets = ["[Google Snippets]: "] + [f"{i+1}. " + x["snippet"] for i,x in enumerate(results)]
            context = "\n".join(snippets)

        SEARCH_DATABASE[query] = context
        with open("data/search_database_serper_new.jsonl", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps({"key": query, "value": context}) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

        return context


def google_retrieve(query, max_retries=3):

    results = google_search(query, n=5)
    all_error = True
    all_cleaned_title_paragraphs = []
    for result in results:
        text = scrape_text(result["url"])
        if text.startswith('Error:'):
            continue
        else:
            all_error = False
        filtered_text = filter_text(text)
        paragraphs = split_into_paragraphs(filtered_text)
        all_cleaned_title_paragraphs.extend([(result['title'], p) for p in paragraphs])

    all_cleaned_paragraphs = [p[1] for p in all_cleaned_title_paragraphs]
    _, idxs = retrieve(query, all_cleaned_paragraphs, 10)
    relevant_title_paragraphs = [all_cleaned_title_paragraphs[i] for i in idxs]
    if all_error:
        return None
    else:
        return relevant_title_paragraphs


def google_retrieve_rank(query):
    title_paragraphs = google_retrieve(query)
    titles = [p[0] for p in title_paragraphs]
    paragraphs = [p[1] for p in title_paragraphs]
    relevant_titles, relevant_paragraphs, _ = rerank(query, titles, paragraphs, 5)

    return relevant_titles, relevant_paragraphs
