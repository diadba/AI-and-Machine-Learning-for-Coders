"""
    Python Interpreter version: 3.11
    TensorFlow version: 2.14.0
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

### Initial Tokenization.
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

# Initial tokenization of the corpus, no OOV used
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
print('-' * 128)
print('word index:', word_index)
print('sequences:', sequences)

### Exploring Test Data with Unseen Words.
test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

test_sequences = tokenizer.texts_to_sequences(test_data)
print('-' * 128)
print('word index:', word_index)
print('test sequences:', test_sequences)

# Invert the dictionary.
index_to_word = {v: k for k, v in word_index.items()}

# Convert sequences.
decoded = [[index_to_word[i] for i in seq] for seq in test_sequences]

for i, seq in enumerate(decoded, start=1):
    print('sequence ', i, ": ", seq, ' from ', (test_data[i-1]).lower(), sep='')

### Adding OOV to improve test data sequences
# Here you can re-tokenize with an OOV token.
tokenizer = Tokenizer(num_words=100, oov_token="")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

test_sequences = tokenizer.texts_to_sequences(test_data)
print('-' * 128)
print('word index:', word_index)
print('test sequences:', test_sequences)

### Explore Padding
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
    'I really enjoyed walking in the snow today'
]

# Re-tokenize with the new sentences from above
tokenizer = Tokenizer(num_words=100, oov_token="")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
print('-' * 128)
print('sequences:', sequences)

padded = pad_sequences(sequences)
print('padded:', padded)

padded = pad_sequences(padded, padding='post')
print('post-padded:', padded)

padded = pad_sequences(padded, padding='post', maxlen=6)
print('post-6-padded:', padded)

padded = pad_sequences(padded, padding='post', maxlen=6, truncating='post')
print('post-6-truncated-padded:', padded)