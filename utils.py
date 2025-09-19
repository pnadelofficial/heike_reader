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
    
    def search_glosses(self, search_term):
        """
        Search for glosses matching the search term across multiple fields.
        Returns matching glosses and information about what type of match was found.
        """
        search_term_lower = search_term.lower()
        
        # Search in multiple fields
        matches = []
        
        # 1. Exact token match (Japanese characters)
        token_matches = self.glosses_df[self.glosses_df['token'] == search_term]
        for _, row in token_matches.iterrows():
            matches.append({
                'row': row,
                'match_type': 'exact_token',
                'match_field': 'token',
                'search_token': row['token']  # The token to highlight in sentences
            })
        
        # 2. Token transliteration match (case-insensitive)
        transliteration_matches = self.glosses_df[
            self.glosses_df['token_transliteration'].str.lower() == search_term_lower
        ]
        for _, row in transliteration_matches.iterrows():
            # Avoid duplicates if already matched by token
            if not any(match['row']['token'] == row['token'] and 
                      match['row']['chapter_id'] == row['chapter_id'] for match in matches):
                matches.append({
                    'row': row,
                    'match_type': 'token_transliteration',
                    'match_field': 'token_transliteration',
                    'search_token': row['token']
                })
        
        # 3. Lemma match (Japanese characters)
        lemma_matches = self.glosses_df[self.glosses_df['lemma'] == search_term]
        for _, row in lemma_matches.iterrows():
            # Avoid duplicates
            if not any(match['row']['token'] == row['token'] and 
                      match['row']['chapter_id'] == row['chapter_id'] for match in matches):
                matches.append({
                    'row': row,
                    'match_type': 'lemma',
                    'match_field': 'lemma',
                    'search_token': row['token']
                })
        
        # 4. Lemma transliteration match (case-insensitive)
        lemma_transliteration_matches = self.glosses_df[
            self.glosses_df['lemma_transliteration'].str.lower() == search_term_lower
        ]
        for _, row in lemma_transliteration_matches.iterrows():
            # Avoid duplicates
            if not any(match['row']['token'] == row['token'] and 
                      match['row']['chapter_id'] == row['chapter_id'] for match in matches):
                matches.append({
                    'row': row,
                    'match_type': 'lemma_transliteration',
                    'match_field': 'lemma_transliteration',
                    'search_token': row['token']
                })
        
        # 5. Partial matches in transliterations (contains search term)
        partial_token_matches = self.glosses_df[
            self.glosses_df['token_transliteration'].str.lower().str.contains(search_term_lower, na=False)
        ]
        for _, row in partial_token_matches.iterrows():
            # Avoid duplicates and exact matches already found
            if not any(match['row']['token'] == row['token'] and 
                      match['row']['chapter_id'] == row['chapter_id'] for match in matches):
                matches.append({
                    'row': row,
                    'match_type': 'partial_token_transliteration',
                    'match_field': 'token_transliteration',
                    'search_token': row['token']
                })
        
        partial_lemma_matches = self.glosses_df[
            self.glosses_df['lemma_transliteration'].str.lower().str.contains(search_term_lower, na=False)
        ]
        for _, row in partial_lemma_matches.iterrows():
            # Avoid duplicates
            if not any(match['row']['token'] == row['token'] and 
                      match['row']['chapter_id'] == row['chapter_id'] for match in matches):
                matches.append({
                    'row': row,
                    'match_type': 'partial_lemma_transliteration',
                    'match_field': 'lemma_transliteration',
                    'search_token': row['token']
                })
        
        return matches
    
    def get_match_summary(self, matches):
        """Generate a summary of what types of matches were found"""
        if not matches:
            return "No matches found."
        
        match_types = {}
        for match in matches:
            match_type = match['match_type']
            if match_type not in match_types:
                match_types[match_type] = 0
            match_types[match_type] += 1
        
        summary_parts = []
        type_descriptions = {
            'exact_token': 'exact token matches',
            'token_transliteration': 'token transliteration matches',
            'lemma': 'lemma matches',
            'lemma_transliteration': 'lemma transliteration matches',
            'partial_token_transliteration': 'partial token transliteration matches',
            'partial_lemma_transliteration': 'partial lemma transliteration matches'
        }
        
        for match_type, count in match_types.items():
            if match_type in type_descriptions:
                summary_parts.append(f"{count} {type_descriptions[match_type]}")
        
        return f"Found {len(matches)} total matches: " + ", ".join(summary_parts)
    
    def search(self, search_term):
        """
        Enhanced search function that searches across multiple fields
        """
        if not search_term.strip():
            st.write("Please enter a search term.")
            return
        
        # Get all matching glosses
        matches = self.search_glosses(search_term.strip())
        
        if not matches:
            st.write(f"No matches found for: '{search_term}'")
            return
        
        # Display search summary
        st.write(self.get_match_summary(matches))
        
        # Group matches by chapter
        chapters_with_matches = {}
        for match in matches:
            chapter_id = match['row']['chapter_id']
            if chapter_id not in chapters_with_matches:
                chapters_with_matches[chapter_id] = []
            chapters_with_matches[chapter_id].append(match)
        
        valid_chapter_ids = list(chapters_with_matches.keys())
        st.write(f"Chapters: {sorted(valid_chapter_ids)}")
        
        # Process each chapter
        for chapter_id in sorted(valid_chapter_ids):
            chapter_matches = chapters_with_matches[chapter_id]
            
            try:
                chapter_text = self.chapters[chapter_id]
            except KeyError:
                st.write(f"⚠️ Chapter {chapter_id} text not available")
                continue
            
            # Get all unique tokens to search for in this chapter
            tokens_to_find = list(set(match['search_token'] for match in chapter_matches))
            
            # Find sentences containing these tokens
            sentences = self.sent_tokenizer.tokenize(chapter_text)
            
            for i, sentence in enumerate(sentences):
                # Check if any of our target tokens appear in this sentence
                sentence_has_match = any(token in sentence for token in tokens_to_find)
                if sentence_has_match:
                    # Find corresponding translation
                    translated_rows = self.translated_df[
                        (self.translated_df['original'].apply(lambda x: x in sentence if x else False)) & 
                        (self.translated_df['chapter_id'] == chapter_id)
                    ]
                    
                    # Display each matching sentence
                    for _, row in translated_rows.iterrows():
                        # Determine which token(s) to highlight in this sentence
                        highlighting_tokens = [token for token in tokens_to_find if token in row['original']]
                        
                        if highlighting_tokens:
                            # Show match information for this sentence
                            relevant_matches = [
                                match for match in chapter_matches 
                                if match['search_token'] in row['original']
                            ]
                            
                            if relevant_matches:
                                st.write(f"#### Chapter {chapter_id}")
                                
                                # Show what type of matches were found in this sentence
                                match_info = []
                                for match in relevant_matches:
                                    match_desc = f"'{match['row'][match['match_field']]}' ({match['match_type'].replace('_', ' ')})"
                                    match_info.append(match_desc)
                                
                                if len(match_info) > 1:
                                    st.write(f"**Matches:** {', '.join(match_info)}")
                                else:
                                    st.write(f"**Match:** {match_info[0]}")
                                
                                # Use the first highlighting token for the display
                                # (the enhanced display function can handle multiple tokens if needed)
                                display_sentence(
                                    row['original'], 
                                    row['translation'], 
                                    self.glosses_df, 
                                    sent_id=f"{chapter_id}_{i}", 
                                    searched_token=highlighting_tokens[0]
                                )
                                
                                break  # Only show each sentence once per chapter

