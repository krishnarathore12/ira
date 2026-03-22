import asyncio
import re
from typing import Awaitable, TypeVar

import nltk
import pendulum
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords

from himem.utils.factory import LlmFactory

DEFAULT_DATE_FORMAT = "YYYY-MM-DD HH:mm"
T = TypeVar("T")

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except Exception:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')


def extract_result(text, tag="tag"):
    pattern = rf"<{tag}>([\s\S]*?)<\/{tag}>"
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        return "", False
    else:
        return matches[0], True


def prefix_exchanges_with_idx(exchanges):
    exchanges_str_with_idx = ""
    for i, exchange in enumerate(exchanges):
        exchanges_str_with_idx += f"[Exchange {i}]: {exchange}\n\n"
    return exchanges_str_with_idx


def is_determiner_nltk(word, sentence):
    """Checks if a word is a determiner within its sentence context using NLTK."""
    tokens = word_tokenize(sentence)
    tagged_words = pos_tag(tokens)

    determiner_tags = ['DT', 'PRP$', 'WDT', 'PDT']  # DT, Possessive Pronoun, Wh-Determiner, Pre-Determiner

    for w, tag in tagged_words:
        if w.lower() == word.lower():
            # Check if the tag belongs to one of the determiner categories
            return tag in determiner_tags
    return False


def convert_to_possessive_determiner(noun):
    """Adds the possessive suffix ('s or ') to a noun."""
    if noun.lower().endswith('s'):
        # For plurals or names ending in 's' (e.g., workers, James)
        return noun + "'"
    else:
        # For most singular nouns (e.g., dog, Alice)
        return noun + "'s"


def find_stopwords(sentence):
    # Convert the sentence to lower case for comparison
    sentence = sentence.lower()

    # Get the official list of English stop words
    stop_words = set(stopwords.words('english'))

    # Tokenize the sentence (split it into individual words)
    words = word_tokenize(sentence)

    # Find the words in the sentence that are also in the stop_words set
    found_stopwords = [word for word in words if word in stop_words]

    return list(set(found_stopwords))


def filter_stopwords(sentence: str):
    stop_words = find_stopwords(sentence)
    for stop_word in stop_words:
        sentence = sentence.replace(stop_word, '')
    return sentence.strip()


def get_absolute_timeline(expr, dt):
    expr_lower = expr.lower()
    if 'year' in expr_lower:
        timeline_format = "YYYY"
    elif 'month' in expr_lower:
        timeline_format = "YYYY-MM"
    elif any(word in expr_lower for word in ['week', 'day', 'yesterday', 'tomorrow', 'today']):
        timeline_format = "YYYY-MM-DD"
    else:
        timeline_format = DEFAULT_DATE_FORMAT
    return pendulum.instance(dt).format(timeline_format)


def run_sync(coro: Awaitable[T]) -> T:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no running loop
        return asyncio.run(coro)
    else:
        # Nested loop allowed in Python 3.7+ with uvloop or standard lib
        return loop.run_until_complete(coro)


def create_llm_instance_from_config(config, enabled_llm_provider):
    llm_config = config.llm_providers[enabled_llm_provider].config
    return LlmFactory.create(enabled_llm_provider, llm_config)
