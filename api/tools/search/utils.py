import re
from urllib.parse import urlparse, urljoin
import requests
import torch
import transformers
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RETRIEVER = None
READER_TOKENIZER = None
READER = None

def vector_search(query_embedding, embeddings, k):
    sim_scores = np.inner(embeddings, query_embedding) / (norm(embeddings, axis=1) * norm(query_embedding))
    max_idxs = np.argpartition(sim_scores, -k)[-k:]
    return max_idxs


def extract_urls(text):
    url_pattern = r'https?://\S+'
    urls = re.findall(url_pattern, text)
    return urls


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# Function to sanitize the URL
def sanitize_url(url):
    return urljoin(url, urlparse(url).path)


def get_response(url, timeout=10):
    try:
        # Most basic check if the URL is valid:
        if not url.startswith('http://') and not url.startswith('https://'):
            raise ValueError('Invalid URL format')

        sanitized_url = sanitize_url(url)
        response = requests.get(sanitized_url, timeout=timeout)

        # Check if the response contains an HTTP error
        if response.status_code >= 400:
            return None, "Error: HTTP " + str(response.status_code) + " error"

        return response, None
    except ValueError as ve:
        # Handle invalid URL format
        return None, "Error: " + str(ve)

    except requests.exceptions.RequestException as re:
        # Handle exceptions related to the HTTP request (e.g., connection errors, timeouts, etc.)
        return None, "Error: " + str(re)


def scrape_text(url):
    """Scrape text from a webpage"""
    response, error_message = get_response(url)
    if error_message:
        return error_message

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def filter_text(text):
    filtered_text = ""
    texts = text.split("\n")
    for t in texts:
        if len(t.split(" ")) > 10:
            filtered_text += t + "\n"
    return filtered_text


def split_into_paragraphs(text):
    paragraphs = []
    raw_paragraphs = text.split("\n")
    for p in raw_paragraphs:
        sub_paragraphs = []
        if len(p.split(" ")) > 100:
            sentences = sent_tokenize(p)
            paragraph = ""
            paragraph_len = 0
            for s in sentences:
                sentence_len = len(s.split(" "))
                paragraph += s + " "
                paragraph_len += sentence_len
                if paragraph_len > 100:
                    sub_paragraphs.append(paragraph)
                    paragraph = ""
                    paragraph_len = 0
        else:
            sub_paragraphs.append(p)

        paragraphs.extend(sub_paragraphs)

    return paragraphs


def retrieve(query, paragraphs, k):
    k = min(k, len(paragraphs))
    global RETRIEVER
    if RETRIEVER is None:
        RETRIEVER = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = RETRIEVER.encode(paragraphs)
    query_embedding = RETRIEVER.encode(query)
    # print(f"All paragraphs: {paragraphs}")
    max_idxs = vector_search(query_embedding, embeddings, k=k)
    relevant_paragraphs = [paragraphs[i] for i in max_idxs]
    # print(f"Retrieved paragraphs: {relevant_paragraphs}")
    return relevant_paragraphs, max_idxs

    # relevant_text = ""
    # relevant_idx = 1
    # relevant_len = 0
    # # if relevant paragraph are shorter than 50 words, we will add more paragraphs
    # for paragraph in relevant_paragraphs:
    #         if relevant_len < 50:
    #             relevant_text += f"Text {relevant_idx}: " + paragraph + " "
    #             relevant_idx += 1
    #             relevant_len += len(paragraph.split(" "))
    # return relevant_text

def get_max_idx(x, k):
    max_num_idx = np.argsort(x)[-k:]
    max_num_idx = max_num_idx[::-1]
    return max_num_idx


def rerank(query, titles, paragraphs, k):
    global READER_TOKENIZER
    global READER
    global DEVICE
    if READER is None:
        READER_TOKENIZER = transformers.DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-multiset-base")
        READER = transformers.DPRReader.from_pretrained("facebook/dpr-reader-multiset-base").to(DEVICE)
    k = min(k, len(paragraphs))
    sim_scores = []
    for i in range(0, len(titles), 10):
        encoded_inputs = READER_TOKENIZER(
            questions=query,
            titles=titles[i: i+10],
            texts=paragraphs[i: i+10],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = READER(**encoded_inputs)
        scores = outputs.relevance_logits.cpu().detach().numpy().tolist()
        sim_scores.extend(scores)

    max_idxs = get_max_idx(sim_scores, k)
    relevant_titles = [titles[i] for i in max_idxs]
    relevant_paragraphs = [paragraphs[i] for i in max_idxs]

    return relevant_titles, relevant_paragraphs, max_idxs