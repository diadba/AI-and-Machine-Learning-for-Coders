"""
    Python Interpreter version: 3.11
    TensorFlow version: 2.14.0
"""
from typing import Any
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

tokenizer=Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index: dict[Any, Any]=tokenizer.word_index
print('First iteration:', word_index)

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

tokenizer=Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index: dict[Any, Any]=tokenizer.word_index
print('Second iteration:', word_index)