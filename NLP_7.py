import random
import re
import nltk
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer
nltk.download('punkt')

class NGramAutoComplete:
    def __init__(self, n=2):
        """
        Initializes the N-Gram Model.
        :param n: Specifies the order of the N-gram model (e.g., 2 for bigram, 3 for trigram)
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()

    def preprocess_text(self, text):
        """
        Cleans and tokenizes input text.
        """
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
        tokens = word_tokenize(text)
        return tokens

    def train(self, text):
        """
        Trains the N-Gram model on input text.
        """
        tokens = self.preprocess_text(text)
        self.vocab.update(tokens)

        # Store unigram counts
        self.unigram_counts.update(tokens)

        # Create n-grams
        n_grams = list(ngrams(tokens, self.n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))
        
        # Count occurrences
        for ngram in n_grams:
            prefix = tuple(ngram[:-1])  # Previous words
            next_word = ngram[-1]       # Next word prediction
            self.ngram_counts[prefix][next_word] += 1

    def predict_next_word(self, text):
        """
        Predicts the most likely next word given a phrase.
        If no exact match is found, it falls back to unigram distribution.
        """
        tokens = self.preprocess_text(text)
        prefix = tuple(tokens[-(self.n - 1):])  # Get the last N-1 words as prefix
        
        # If prefix exists in n-grams, predict based on probability
        if prefix in self.ngram_counts:
            probable_words = self.ngram_counts[prefix]
            total_count = sum(probable_words.values())
            choices, weights = zip(*[(word, count / total_count) for word, count in probable_words.items()])
            return random.choices(choices, weights=weights)[0]

        # Fallback: Predict based on unigram probabilities
        elif self.unigram_counts:
            choices, weights = zip(*[(word, count / sum(self.unigram_counts.values())) for word, count in self.unigram_counts.items()])
            return random.choices(choices, weights=weights)[0]

        # Last fallback: Random word from vocabulary
        return random.choice(list(self.vocab)) if self.vocab else "No prediction available"

# Example Usage
if __name__ == "__main__":
    text_corpus = """
    Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.
    Auto-complete is a feature that predicts words as a user types in a search box or messaging app.
    The n-gram model is commonly used in NLP for text prediction.
    """

    model = NGramAutoComplete(n=2)  # Bigram Model
    model.train(text_corpus)

    # Test Auto-complete
    test_sentences = ["Natural Language", "Auto-complete is", "The n-gram", "NLP is a", "Helps computers"]

    for sentence in test_sentences:
        predicted_word = model.predict_next_word(sentence)
        print(f"Input: '{sentence}' → Predicted Next Word: '{predicted_word}'")
