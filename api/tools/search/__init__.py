from .google_search import google_retrieve_rank
from .utils import rerank, retrieve


def GOOGLESearch(query):
    relevant_titles, relevant_paragraphs = google_retrieve_rank(query)

    return relevant_paragraphs[0]




