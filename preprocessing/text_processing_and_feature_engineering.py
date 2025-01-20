# -*- coding: utf-8 -*-

import re
import string
import nltk
from nltk.corpus import stopwords
import spacy

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"\d+", "<NUMBER>", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word.lower() not in stop_words)

def calculate_avg_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0

def calculate_sentence_count(text):
    return text.count('.') + text.count('!') + text.count('?')

def symbol_ratio(text, symbol):
    return text.count(symbol) / len(text) if len(text) > 0 else 0

def symbol_to_word_ratio(text):
    total_words = len(text.split())
    symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return symbol_count / total_words if total_words > 0 else 0

def count_stop_words(text):
    if not isinstance(text, str):
        return 0
    tokens = text.translate(str.maketrans('', '', string.punctuation)).split()
    return sum(1 for word in tokens if word.lower() in stop_words)

def preprocess_and_engineer_features(dataset):
    dataset["text"] = dataset["text"].str.strip()
    dataset["title"] = dataset["title"].str.strip()
    dataset = dataset[(dataset["text"] != "") & (dataset["title"] != "")]
    dataset["cleaned_text"] = dataset["text"].apply(normalize_text)
    dataset["cleaned_title"] = dataset["title"].apply(normalize_text)

    dataset["cleaned_and_without_stop_text"] = dataset["cleaned_text"].apply(remove_stopwords)
    dataset["cleaned_and_without_stop_title"] = dataset["cleaned_title"].apply(remove_stopwords)

    dataset["char_count"] = dataset["text"].apply(len)
    dataset["word_count"] = dataset["text"].apply(lambda x: len(x.split()))
    dataset["unique_word_count"] = dataset["text"].apply(lambda x: len(set(x.split())))
    dataset["avg_word_length"] = dataset["text"].apply(calculate_avg_word_length)
    dataset["sentence_count"] = dataset["text"].apply(calculate_sentence_count)
    dataset["symbol_to_word_ratio"] = dataset["text"].apply(symbol_to_word_ratio)
    dataset["capital_letter_ratio"] = dataset["text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    dataset["special_char_ratio"] = dataset["text"].apply(
        lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / len(x) if len(x) > 0 else 0
    )
    dataset["comma_ratio"] = dataset["text"].apply(lambda x: x.count(",") / len(x) if len(x) > 0 else 0)
    dataset["period_ratio"] = dataset["text"].apply(lambda x: x.count(".") / len(x) if len(x) > 0 else 0)
    dataset["question_mark_ratio"] = dataset["text"].apply(lambda x: x.count("?") / len(x) if len(x) > 0 else 0)
    dataset["exclamation_mark_ratio"] = dataset["text"].apply(lambda x: x.count("!") / len(x) if len(x) > 0 else 0)
    dataset["ellipses_ratio"] = dataset["text"].apply(lambda x: x.count("...") / len(x) if len(x) > 0 else 0)
    dataset["url_count_ratio"] = dataset["text"].apply(lambda x: (x.count("http") + x.count("www")) / len(x) if len(x) > 0 else 0)
    dataset["hashtag_ratio"] = dataset["text"].apply(lambda x: symbol_ratio(x, "#"))
    dataset["at_ratio"] = dataset["text"].apply(lambda x: symbol_ratio(x, "@"))
    dataset["quote_ratio"] = dataset["text"].apply(lambda x: symbol_ratio(x, '"'))
    dataset["title_char_count"] = dataset["title"].apply(len)
    dataset["title_word_count"] = dataset["title"].apply(lambda x: len(x.split()))
    dataset["title_unique_word_count"] = dataset["title"].apply(lambda x: len(set(x.split())))
    dataset["avg_word_length_title"] = dataset["title"].apply(calculate_avg_word_length)
    dataset["stop_word_count_title"] = dataset["title"].apply(count_stop_words)
    dataset["stop_word_ratio_title"] = dataset["stop_word_count_title"] / dataset["title_word_count"].replace(0, 1)
    dataset["cleaned_word_lengths"] = dataset["cleaned_and_without_stop_text"].apply(lambda x: [len(word) for word in x.split()])
    dataset["cleaned_title_word_lengths"] = dataset["cleaned_and_without_stop_title"].apply(lambda x: [len(word) for word in x.split()])

    return dataset