import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dataclasses import dataclass
from konoha import SentenceTokenizer
from tokenizer import ClassicalJapaneseTokenizer
import html
import json

@st.cache_resource
def get_tokenizer():
    tokenizer = ClassicalJapaneseTokenizer()
    tokenizer.load_model("heike_tokenizer")
    return tokenizer

@st.cache_resource
def get_sentence_tokenizer():
    sent_tokenizer = SentenceTokenizer()
    return sent_tokenizer

@st.cache_data
def load_data():
    glosses = pd.read_csv("data/heike_token_level_glosses.csv")
    translated = pd.read_csv("data/heike_sentence_level_translations_updated912.csv").dropna().reset_index(drop=True)
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
                    transliteration=row['token_transliteration'] if isinstance(row['token_transliteration'], str) else "N/A",
                    lemma=row['lemma'] if isinstance(row['lemma'], str) else "N/A",
                    lemma_transliteration=row['lemma_transliteration'] if isinstance(row['lemma_transliteration'], str) else "N/A",
                    part_of_speech=row['token_part_of_speech'] if isinstance(row['token_part_of_speech'], str) else "N/A",
                    gloss=row['gloss'] if isinstance(row['gloss'], str) else "N/A"
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

def display_sentence(sentence, translation, glosses_df, sent_id, searched_token=None):
    """
    Enhanced display_sentence function using HTML/CSS/JavaScript for better performance
    and copyable annotations.
    """
    
    # Create HeikeSentence object and tokenize
    heike_sentence = HeikeSentence(sentence, translation)
    heike_sentence.tokenize()
    heike_sentence.annotate_tokens(glosses_df)
    
    # Prepare token data for HTML
    tokens_data = []
    for i, (token, annotation) in enumerate(heike_sentence.annotations):
        # Escape HTML characters
        token_clean = html.escape(token)
     
        # Prepare annotation data
        annotation_html = f"""
        <div class="annotation-header">Token Information</div>
        <div class="annotation-field"><strong>Token:</strong> {html.escape(annotation.token)}</div>
        <div class="annotation-field"><strong>Transliteration:</strong> {html.escape(annotation.transliteration)}</div>
        <div class="annotation-field"><strong>Lemma:</strong> {html.escape(annotation.lemma)} ({html.escape(annotation.lemma_transliteration)})</div>
        <div class="annotation-field"><strong>Part of Speech:</strong> {html.escape(annotation.part_of_speech)}</div>
        <div class="annotation-field"><strong>Gloss:</strong> {html.escape(annotation.gloss)}</div>
        """
        
        # Check if this token contains the searched term
        is_searched = searched_token and searched_token in token
        
        tokens_data.append({
            'token': token_clean,
            'annotation_html': annotation_html,
            'is_searched': is_searched,
            'index': i
        })
    
    # Generate unique IDs for this sentence
    sentence_id = f"sentence_{sent_id}"
    
    # Build the complete HTML document
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
        body {{
            margin: 0;
            padding: 15px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        
        .sentence-container {{
            margin: 0;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        
        .tokens-container {{
            line-height: 2.2;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        
        .token {{
            display: inline-block;
            padding: 4px 6px;
            margin: 2px 1px;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s ease;
            background-color: #f8f9fa;
            border: 1px solid transparent;
            user-select: none;
        }}
        
        .token:hover {{
            background-color: #e9ecef;
            border-color: #dee2e6;
            transform: translateY(-1px);
        }}
        
        .token.selected {{
            background-color: #007bff;
            color: white;
            border-color: #0056b3;
        }}
        
        .token.searched {{
            background-color: #fff3cd;
            border-color: #ffeaa7;
            font-weight: bold;
        }}
        
        .token.searched:hover {{
            background-color: #fff3cd;
            border-color: #f39c12;
        }}
        
        .annotation-area {{
            margin: 15px 0;
            padding: 15px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: none;
            animation: slideDown 0.3s ease;
        }}
        
        .annotation-area.show {{
            display: block;
        }}
        
        @keyframes slideDown {{
            from {{
                opacity: 0;
                max-height: 0;
                padding: 0 15px;
            }}
            to {{
                opacity: 1;
                max-height: 300px;
                padding: 15px;
            }}
        }}
        
        .annotation-header {{
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }}
        
        .annotation-field {{
            margin: 8px 0;
            font-size: 14px;
            color: #555;
        }}
        
        .annotation-field strong {{
            color: #333;
            min-width: 120px;
            display: inline-block;
        }}
        
        .translation {{
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 14px;
            color: #666;
        }}
        
        .translation strong {{
            color: #333;
        }}
        </style>
    </head>
    <body>
        <div class="sentence-container" id="{sentence_id}">
            <div class="tokens-container">
                {' '.join([
                    f'<span class="token {"searched" if token_data["is_searched"] else ""}" '
                    f'data-token-index="{token_data["index"]}" '
                    f'data-sentence-id="{sentence_id}" '
                    f'onclick="toggleAnnotation(\'{sentence_id}\', {token_data["index"]})">'
                    f'{token_data["token"]}'
                    f'</span>'
                    for token_data in tokens_data
                ])}
            </div>
            
            <div class="annotation-area" id="{sentence_id}_annotation">
                <div class="annotation-content" id="{sentence_id}_content">
                    <!-- Annotation content will be populated by JavaScript -->
                </div>
            </div>
            
            <div class="translation">
                <strong>Translation:</strong> {html.escape(translation)}
            </div>
        </div>

        <script>
        // Store annotation data for each sentence
        const annotationData = {json.dumps([token_data['annotation_html'] for token_data in tokens_data])};
        
        function toggleAnnotation(sentenceId, tokenIndex) {{
            const annotationArea = document.getElementById(sentenceId + '_annotation');
            const annotationContent = document.getElementById(sentenceId + '_content');
            const tokens = document.querySelectorAll(`[data-sentence-id="${{sentenceId}}"]`);
            const clickedToken = document.querySelector(`[data-sentence-id="${{sentenceId}}"][data-token-index="${{tokenIndex}}"]`);
            
            // Remove selected class from all tokens in this sentence
            tokens.forEach(token => token.classList.remove('selected'));
            
            // If clicking the same token that's already showing, hide annotation
            if (annotationArea.classList.contains('show') && clickedToken.classList.contains('was-selected')) {{
                annotationArea.classList.remove('show');
                clickedToken.classList.remove('was-selected');
                return;
            }}
            
            // Clear was-selected from all tokens
            tokens.forEach(token => token.classList.remove('was-selected'));
            
            // Show annotation for clicked token
            clickedToken.classList.add('selected', 'was-selected');
            annotationContent.innerHTML = annotationData[tokenIndex];
            annotationArea.classList.add('show');
        }}
        </script>
    </body>
    </html>
    """
    
    # Render using components.html which properly handles JavaScript
    components.html(html_content, height=300, scrolling=True)
    
    # Add separator after the component
    st.markdown("---")

class Searcher:
    def __init__(self, translated_df, glosses_df):
        self.translated_df = translated_df
        self.chapters = self.translated_df.groupby('chapter_id')['original'].apply(lambda texts: ''.join(texts)).to_dict()
        self.glosses_df = glosses_df
        self.tokenizer = get_tokenizer()
        self.sent_tokenizer = get_sentence_tokenizer()
    
    def search(self, token):
        glosses = self.glosses_df[(self.glosses_df['token'] == token) | (self.glosses_df['token_transliteration'] == token)]
        if glosses.empty:
            st.write(f"No gloss found for token: {token}")
            return
        valid_chapter_ids = glosses['chapter_id'].unique()
        self.len_results = len(glosses)
        st.write(f"Found {self.len_results} sentences containing '{token}' in chapters: {valid_chapter_ids}")
        for chapter_id in valid_chapter_ids:
            try:
                chapter_text = self.chapters[chapter_id] # missing chapter 88, need to redo translation from this point
            except Exception as e:
                print(f"Error retrieving chapter {chapter_id}: {e}")
                continue
            sentences = self.sent_tokenizer.tokenize(chapter_text)
            for i, sentence in enumerate(sentences):
                if token in sentence:
                    translated = self.translated_df[(self.translated_df['original'].apply(lambda x: True if x in sentence else False)) & (self.translated_df['chapter_id'] == chapter_id)]
                    st.write(f"#### Chapter {chapter_id}")
                    for _, row in translated.iterrows():
                        if token in row['original']:
                            display_sentence(row['original'], row['translation'], self.glosses_df, sent_id=i, searched_token=token)
