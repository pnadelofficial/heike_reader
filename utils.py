import streamlit as st
import pandas as pd
from dataclasses import dataclass
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import re
from collections import defaultdict
import pickle
import json

class ClassicalJapaneseTokenizer:
    def __init__(self):
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,  # L1 regularization coefficient
            c2=0.1,  # L2 regularization coefficient
            max_iterations=100,
            all_possible_transitions=True
        )
        self.char_stats = defaultdict(int)

    def get_char_type(self, char):
        """Classify character types - important for Japanese text"""
        if re.match(r'[\u3040-\u309F]', char):  # Hiragana
            return 'HIRAGANA'
        elif re.match(r'[\u30A0-\u30FF]', char):  # Katakana
            return 'KATAKANA'
        elif re.match(r'[\u4E00-\u9FAF]', char):  # Kanji
            return 'KANJI'
        elif re.match(r'[a-zA-Z]', char):
            return 'LATIN'
        elif re.match(r'[0-9]', char):
            return 'DIGIT'
        elif char in '。、？！':  # Common punctuation
            return 'PUNCT'
        else:
            return 'OTHER'
    
    def char_features(self, text, i):
        """
        Extract features for character at position i.
        This is where the magic happens - we tell the CRF what to pay attention to.
        """
        char = text[i]
        features = {
            # Basic character information
            'char': char,
            'char_type': self.get_char_type(char),
            
            # Position in text
            'position': i,
            'is_start': i == 0,
            'is_end': i == len(text) - 1,
        }

        # Context features - look at surrounding characters
        # These help the model learn patterns like "particle の often follows kanji"
        if i > 0:
            features['char-1'] = text[i-1]
            features['char_type-1'] = self.get_char_type(text[i-1])
            
            # Transition features (very important for Japanese)
            features['transition'] = f"{self.get_char_type(text[i-1])}→{self.get_char_type(char)}"
            
        if i > 1:
            features['char-2'] = text[i-2]
            features['char_type-2'] = self.get_char_type(text[i-2])
            
        if i < len(text) - 1:
            features['char+1'] = text[i+1]
            features['char_type+1'] = self.get_char_type(text[i+1])
            
        if i < len(text) - 2:
            features['char+2'] = text[i+2]
            features['char_type+2'] = self.get_char_type(text[i+2])
            
        # Bigram and trigram features
        if i > 0:
            features['bigram-1'] = text[i-1:i+1]
        if i > 0 and i < len(text) - 1:
            features['trigram'] = text[i-1:i+2]
            
        # Frequency-based features (learned from training data)
        if char in self.char_stats:
            features['char_freq'] = 'high' if self.char_stats[char] > 10 else 'low'
            
        return features

    def sentence_to_features(self, text):
        """Convert a sentence (string) to a list of feature dictionaries"""
        return [self.char_features(text, i) for i in range(len(text))]

    def create_boundary_labels(self, tokens):
        """
        Convert a list of tokens back to character-level boundary labels.
        
        For example: ['今日', 'は', '良い', '天気'] becomes:
        - Text: "今日は良い天気" 
        - Labels: ['B', 'I', 'B', 'B', 'I', 'B', 'I']
        
        Where B = beginning of word, I = inside word
        """
        full_text = ''.join(tokens)
        labels = []
        char_pos = 0
        
        for token in tokens:
            # First character of each token is beginning
            labels.append('B')
            # Remaining characters are inside
            for _ in range(len(token) - 1):
                labels.append('I')
            char_pos += len(token)
                
        return full_text, labels

    def prepare_training_data(self, treebanked_sentences):
        """
        Convert your treebanked sentences into CRF training format.
        
        Input: List of lists, where each inner list is tokens for one sentence
        Example: [['今日', 'は', '良い', '天気', 'です'], ['彼', 'は', '学生', 'です']]
        """
        X_features = []  # Feature sequences
        y_labels = []    # Label sequences
        
        # Collect character statistics for frequency features
        for sentence_tokens in treebanked_sentences:
            text = ''.join(sentence_tokens)
            for char in text:
                self.char_stats[char] += 1
        
        print(f"Processing {len(treebanked_sentences)} sentences...")
        
        for sentence_tokens in treebanked_sentences:
            # Convert tokens to character sequence with labels
            text, labels = self.create_boundary_labels(sentence_tokens)
            
            # Extract features for each character
            features = self.sentence_to_features(text)
            
            X_features.append(features)
            y_labels.append(labels)
            
        return X_features, y_labels

    def train(self, treebanked_sentences):
        """Train the CRF model on your treebanked data"""
        print("Preparing training data...")
        X_train, y_train = self.prepare_training_data(treebanked_sentences)
        
        print("Training CRF model...")
        self.crf.fit(X_train, y_train)
        
        # Print some training info
        print("Training completed!")
        print(f"Number of features: {len(self.crf.classes_)}")
        print(f"Model classes: {self.crf.classes_}")

    def tokenize(self, text):
        """
        Tokenize new text using the trained model.
        
        Input: Raw text string
        Output: List of tokens
        """
        if not hasattr(self.crf, 'classes_'):
            raise ValueError("Model not trained yet! Call train() first.")
            
        # Extract features for the text
        features = self.sentence_to_features(text)
        
        # Predict labels
        predicted_labels = self.crf.predict([features])[0]
        
        # Convert labels back to tokens
        tokens = []
        current_token = ""
        
        for char, label in zip(text, predicted_labels):
            if label == 'B':  # Beginning of new word
                if current_token:  # Save previous token
                    tokens.append(current_token)
                current_token = char  # Start new token
            else:  # label == 'I' (inside word)
                current_token += char
                
        # Don't forget the last token
        if current_token:
            tokens.append(current_token)
            
        return tokens

    def evaluate(self, test_sentences):
        """Evaluate the model on test data"""
        X_test, y_test = self.prepare_training_data(test_sentences)
        y_pred = self.crf.predict(X_test)
        
        # Calculate metrics
        print("Evaluation Results:")
        print(metrics.flat_classification_report(y_test, y_pred))
        
        return metrics.flat_f1_score(y_test, y_pred, average='weighted')
    
    def save_model(self, filepath):
        """
        Save the trained model and character statistics to disk.
        
        This saves two files:
        1. {filepath}.pkl - The trained CRF model
        2. {filepath}_stats.json - Character frequency statistics
        """
        if not hasattr(self.crf, 'classes_'):
            raise ValueError("Model not trained yet! Cannot save untrained model.")
            
        # Save the CRF model
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.crf, f)
            
        # Save character statistics (needed for frequency features)
        with open(f"{filepath}_stats.json", 'w', encoding='utf-8') as f:
            json.dump(dict(self.char_stats), f, ensure_ascii=False, indent=2)
            
        print(f"Model saved to {filepath}.pkl")
        print(f"Character stats saved to {filepath}_stats.json")
    
    def load_model(self, filepath):
        """
        Load a previously trained model from disk.
        
        Input: filepath (without extension)
        Loads both the model and character statistics
        """
        try:
            # Load the CRF model
            with open(f"{filepath}.pkl", 'rb') as f:
                self.crf = pickle.load(f)
                
            # Load character statistics
            with open(f"{filepath}_stats.json", 'r', encoding='utf-8') as f:
                char_stats_dict = json.load(f)
                self.char_stats = defaultdict(int, char_stats_dict)
                
            print(f"Model loaded from {filepath}.pkl")
            print(f"Character stats loaded from {filepath}_stats.json")
            print(f"Model classes: {self.crf.classes_}")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find model files. Make sure both {filepath}.pkl and {filepath}_stats.json exist.")
            raise e
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

@st.cache_resource
def get_tokenizer():
    tokenizer = ClassicalJapaneseTokenizer()
    tokenizer.load_model("heike_tokenizer")
    return tokenizer

@st.cache_data
def load_data():
    glosses = pd.read_csv("data/heike_token_level_glosses.csv")
    translated = pd.read_csv("data/translated_heike527_qwen14b.csv")
    return glosses, translated

@dataclass
class HeikeToken:
    token: str
    transliteration: str
    lemma: str
    lemma_transliteration: str
    part_of_speech: str
    gloss: str

class HeikeSentence:
    def __init__(self, original, translation):
        self.original = original
        self.translation = translation
        self.tokenizer = get_tokenizer()

    def tokenize(self):
        self.tokens = self.tokenizer.tokenize(self.original)
    
    def annotate_tokens(self, glosses_df):
        self.heike_tokens = []
        for token in self.tokens:
            match = glosses_df[glosses_df['token'] == token]
            if not match.empty:
                row = match.iloc[0] # problem if multiple matches
                heike_token = HeikeToken(
                    token=row['token'],
                    transliteration=row['token_transliteration'],
                    lemma=row['lemma'],
                    lemma_transliteration=row['lemma_transliteration'],
                    part_of_speech=row['token_part_of_speech'],
                    gloss=row['gloss']
                )
            else:
                heike_token = HeikeToken(
                    token=token,
                    transliteration="N/A",
                    lemma="N/A",
                    lemma_transliteration="N/A",
                    part_of_speech="N/A",
                    gloss="N/A"
                )
            self.heike_tokens.append(heike_token)
        self.annotations = zip(self.tokens, self.heike_tokens)
    
tt_css = """
<style>
.token-container {
    font-family: inherit;
    line-height: 1.6;
    margin: 10px 0;
}
.tooltip {
    position: relative;
    display: inline;
    cursor: pointer;
    color: #0066cc;
    font-weight: bold;
    margin-right: 6px;
    padding: 2px 4px;
    border-radius: 3px;
    transition: background-color 0.2s;
}

.tooltip:hover {
    background-color: #f0f8ff;
}

.tooltip:before {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background-color: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    white-space: pre-line;
    z-index: 1000;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s;
    min-width: 200px;
    max-width: 300px;
    text-align: left;
    font-weight: normal;
}

.tooltip:hover:before {
    opacity: 1;
}
</style>
"""

def display_sentence(sentence, translation, glosses_df):
    heike_sentence = HeikeSentence(sentence, translation)
    heike_sentence.tokenize()
    heike_sentence.annotate_tokens(glosses_df)

    html_content = tt_css + '<div class="token-container"><strong>Original:</strong><br>'
    for token, annotation in heike_sentence.annotations:
        tooltip_text = f"{token}\nTransliteration: {annotation.transliteration}\nLemma: {annotation.lemma} ({annotation.lemma_transliteration})\nPart of Speech: {annotation.part_of_speech}\nGloss: {annotation.gloss}"
        html_content += f'<span class="tooltip" data-tooltip="{tooltip_text}">{token}</span>'
    
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

    st.write(f"**Translation:** {heike_sentence.translation}")

class Searcher:
    def __init__(self, translated_df, glosses_df):
        self.translated_df = translated_df
        self.chapters = self.translated_df.groupby('chapter_id')['original'].apply(lambda texts: '。'.join(texts)).to_dict()
        self.glosses_df = glosses_df
        self.tokenizer = get_tokenizer()
    
    def get_context(self, chapter_text, token):
        sentences = chapter_text.split('。')
        context_sentences = []
        for sentence in sentences:
            if token in sentence:
                context_sentences.append(sentence)
        return '。'.join(context_sentences) + '。' if context_sentences else ''
    
    def search(self, token):
        glosses = self.glosses_df[(self.glosses_df['token'] == token) | (self.glosses_df['token_transliteration'] == token)]
        if glosses.empty:
            st.write(f"No gloss found for token: {token}")
            return
        valid_chapter_ids = glosses['chapter_id'].unique()
        self.len_results = len(glosses)
        st.write(f"Found {self.len_results} sentences containing '{token}' in chapters: {valid_chapter_ids}")
        for chapter_id in valid_chapter_ids:
            chapter_text = self.chapters.get(chapter_id, "")
            true_token = glosses[glosses['chapter_id'] == chapter_id]['token'].values[0]
            context = self.get_context(chapter_text, true_token)
            gloss_for_chapter = glosses[glosses['chapter_id'] == chapter_id]
            if context:
                st.markdown(f"### Chapter {chapter_id}")
                st.write(context)
                st.write(f"- {gloss_for_chapter['gloss'].values[0]} (Lemma: {gloss_for_chapter['lemma'].values[0]}, POS: {gloss_for_chapter['token_part_of_speech'].values[0]})")
                st.markdown("---")
            
