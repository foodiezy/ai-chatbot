import csv
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def normalize_text(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    tokens = word_tokenize(text.lower().translate(remove_punct_dict))
    return lemmatize_tokens(tokens)

class SimilarityQA:
    def __init__(self, csv_filepath):
        self.questions = []
        self.answers = []
        self._load_data(csv_filepath)
        

        self.vectorizer = TfidfVectorizer(tokenizer=normalize_text, stop_words='english', token_pattern=None)
        

        if self.questions:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def _load_data(self, filepath):
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) == 2:
                        self.questions.append(row[0])
                        self.answers.append(row[1])
        except Exception as e:
            print(f"Error loading CSV data: {e}")

    def get_best_match(self, user_input, threshold=0.2):
        if not self.questions:
            return None

        user_tfidf = self.vectorizer.transform([user_input])
        cosine_sims = cosine_similarity(user_tfidf, self.tfidf_matrix)[0]
        
        best_idx = cosine_sims.argmax()
        best_score = cosine_sims[best_idx]
        
        if best_score > threshold:
            return self.answers[best_idx]
        else:
            return None

if __name__ == "__main__":
    qa = SimilarityQA("animals_qa.csv")
    print("Test match for 'how long do bad dogs live?' ->", qa.get_best_match("how long do bad dogs live?"))
