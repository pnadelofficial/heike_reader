from utils import get_tokenizer, get_sentence_tokenizer, display_sentence
import streamlit as st
from frequency_analyzer import FrequencyAnalyzer

class Searcher:
    def __init__(self, translated_df, glosses_df):
        self.translated_df = translated_df
        self.chapters = self.translated_df.groupby('chapter_id')['original'].apply(lambda texts: ''.join(texts)).to_dict()
        self.glosses_df = glosses_df
        self.tokenizer = get_tokenizer()
        self.sent_tokenizer = get_sentence_tokenizer()
        self.frequency_analyzer = FrequencyAnalyzer(glosses_df)
    
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

        with st.expander("Frequency Statistics"):
            freq_display = self.frequency_analyzer.format_frequency_display(
                search_term, 'token'
            )
            st.write(freq_display)
            
            chart_data = self.frequency_analyzer.create_frequency_chart_data(
                search_term, 'token'
            )
            if not chart_data.empty:
                st.bar_chart(chart_data)
        
        # Group matches by chapter
        chapters_with_matches = {}
        for match in matches:
            chapter_id = match['row']['chapter_id']
            if chapter_id not in chapters_with_matches:
                chapters_with_matches[chapter_id] = []
            chapters_with_matches[chapter_id].append(match)
        
        valid_chapter_ids = list(chapters_with_matches.keys())
        
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
        
