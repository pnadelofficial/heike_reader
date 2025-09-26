import pandas as pd
import pickle

class FrequencyAnalyzer:
    def __init__(self, glosses_df):
        self.glosses_df = glosses_df
        self._precomputed_stats = {}
        self._precompute_stats()

    def _precompute_stats(self):
        self._precomputed_stats = pickle.load(open("data/precomputed_stats.pkl", "rb"))
    
    def get_term_frequency(self, term, search_type='token'):
        """
        Get frequency information for a specific term
        
        Args:
            term: The term to analyze
            search_type: 'token', 'lemma', 'token_transliteration', or 'lemma_transliteration'
        
        Returns:
            dict with frequency information
        """
        # Handle different search types
        if search_type == 'token':
            matches = self.glosses_df[self.glosses_df['token'] == term]
            frequency = self._precomputed_stats['token_frequencies'].get(term, 0)
            ranking = self._precomputed_stats['token_rankings'].get(term, None)
            total_unique = self._precomputed_stats['unique_tokens']
            lemma = ""
            pos_lemma = self.glosses_df[self.glosses_df['token'] == term]
            if not pos_lemma.empty:
                lemma = pos_lemma['lemma'].iloc[0]
            lemma_pos = self._precomputed_stats['lemma_pos_distribution'].get(lemma, {})
        
        elif search_type == 'lemma':
            matches = self.glosses_df[self.glosses_df['lemma'] == term]
            frequency = self._precomputed_stats['lemma_frequencies'].get(term, 0)
            ranking = self._precomputed_stats['lemma_rankings'].get(term, None)
            total_unique = self._precomputed_stats['unique_lemmas']
            lemma_pos = self._precomputed_stats['lemma_pos_distribution'].get(lemma, {})
        
        elif search_type == 'token_transliteration':
            matches = self.glosses_df[self.glosses_df['token_transliteration'].str.lower() == term.lower()]
            frequency = len(matches)
            ranking = None  # Not pre-computed for transliterations
            total_unique = self.glosses_df['token_transliteration'].nunique()
        
        elif search_type == 'lemma_transliteration':
            matches = self.glosses_df[self.glosses_df['lemma_transliteration'].str.lower() == term.lower()]
            frequency = len(matches)
            ranking = None  # Not pre-computed for transliterations
            total_unique = self.glosses_df['lemma_transliteration'].nunique()
        
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
        
        if frequency == 0:
            return {
                'frequency': 0,
                'ranking': None,
                'total_unique': total_unique,
                'chapters': [],
                'percentage': 0.0
            }
        
        # Chapter distribution
        chapters_with_counts = matches['chapter_id'].value_counts().sort_index().to_dict()
        
        return {
            'frequency': frequency,
            'ranking': ranking,
            'total_unique': total_unique,
            'chapters': chapters_with_counts,
            'percentage': (frequency / self._precomputed_stats['total_tokens']) * 100,
            'unique_chapters': len(chapters_with_counts),
            'lemma_pos': lemma_pos
        }

    def get_form_variations(self, lemma):
        """
        Get all surface forms (tokens) for a given lemma
        
        Returns:
            dict: {surface_form: frequency}
        """
        lemma_matches = self.glosses_df[self.glosses_df['lemma'] == lemma]
        surface_forms = lemma_matches['token'].value_counts().to_dict()
        return surface_forms

    def get_pos_distribution_for_term(self, term, search_type='token'):
        """
        Get part-of-speech distribution for a specific term
        """
        if search_type == 'token':
            matches = self.glosses_df[self.glosses_df['token'] == term]
        elif search_type == 'lemma':
            matches = self.glosses_df[self.glosses_df['lemma'] == term]
        else:
            # For transliterations, find the corresponding tokens first
            if search_type == 'token_transliteration':
                matches = self.glosses_df[self.glosses_df['token_transliteration'].str.lower() == term.lower()]
            else:  # lemma_transliteration
                matches = self.glosses_df[self.glosses_df['lemma_transliteration'].str.lower() == term.lower()]
        
        pos_counts = matches['token_part_of_speech'].value_counts().to_dict()
        return pos_counts
    
    def get_chapter_distribution(self, term, search_type='token'):
        """
        Get detailed chapter distribution for a term
        
        Returns:
            dict with chapter statistics
        """
        frequency_info = self.get_term_frequency(term, search_type)
        chapters = frequency_info['chapters']
        
        if not chapters:
            return {'chapters': {}, 'max_chapter': None, 'min_chapter': None, 'avg_per_chapter': 0}
        
        max_chapter = max(chapters.items(), key=lambda x: x[1])
        min_chapter = min(chapters.items(), key=lambda x: x[1])
        avg_per_chapter = sum(chapters.values()) / len(chapters)
        
        return {
            'chapters': chapters,
            'max_chapter': max_chapter,  # (chapter_id, count)
            'min_chapter': min_chapter,  # (chapter_id, count)
            'avg_per_chapter': avg_per_chapter,
            'total_chapters_with_term': len(chapters)
        }
    
    def get_corpus_summary(self):
        """Get overall corpus statistics"""
        return {
            'total_tokens': self._precomputed_stats['total_tokens'],
            'unique_tokens': self._precomputed_stats['unique_tokens'],
            'unique_lemmas': self._precomputed_stats['unique_lemmas'],
            'pos_distribution': self._precomputed_stats['pos_distribution'],
            'most_frequent_tokens': dict(list(self._precomputed_stats['token_frequencies'].items())[:10]),
            'most_frequent_lemmas': dict(list(self._precomputed_stats['lemma_frequencies'].items())[:10])
        }
    
    def format_frequency_display(self, term, search_type='token'):
        """
        Format frequency information for display in Streamlit
        
        Returns:
            Formatted string ready for st.write() or st.markdown()
        """
        freq_info = self.get_term_frequency(term, search_type)
        
        if freq_info['frequency'] == 0:
            return f"**'{term}'** not found in corpus"
        
        # Build the display string
        parts = []
        
        # Basic frequency
        parts.append(f"**Frequency:** {freq_info['frequency']:,} occurrences")
        
        # Ranking (if available)
        if freq_info['ranking']:
            parts.append(f"**Ranking:** #{freq_info['ranking']:,} of {freq_info['total_unique']:,}")
        
        # Percentage
        parts.append(f"**Percentage:** {freq_info['percentage']:.3f}% of all tokens")
        
        # Chapter distribution
        if freq_info['chapters']:
            chapter_list = sorted(freq_info['chapters'].keys())
            if len(chapter_list) <= 5:
                chapter_str = ", ".join(str(ch) for ch in chapter_list)
            else:
                chapter_str = f"{', '.join(str(ch) for ch in chapter_list[:3])}, ... +{len(chapter_list)-3} more"
            
            parts.append(f"**Chapters:** {chapter_str} ({freq_info['unique_chapters']} total)")
        
        # Lemma POS distribution (if available)
        lemma_poses = []
        if 'lemma_pos' in freq_info and freq_info['lemma_pos']:
            pos_parts = [f"{pos} ({count})" for pos, count in freq_info['lemma_pos'].items()]
            lemma_poses.append("**Part of Speech Distribution:** " + ", ".join(pos_parts))

        return " | ".join(parts) + "\n\n" + " | ".join(lemma_poses) if lemma_poses else ""
    
    def create_frequency_chart_data(self, term, search_type='token'):
        """
        Prepare data for plotting frequency distribution across chapters
        
        Returns:
            pandas DataFrame suitable for st.bar_chart() or other plotting
        """
        freq_info = self.get_term_frequency(term, search_type)
        chapters = freq_info['chapters']
        
        if not chapters:
            return pd.DataFrame()
        
        # Create a complete range of chapters (fill in zeros for missing chapters)
        all_chapters = range(min(chapters.keys()), max(chapters.keys()) + 1)
        chart_data = pd.DataFrame({
            'Chapter': list(all_chapters),
            'Frequency': [chapters.get(ch, 0) for ch in all_chapters]
        })
        
        return chart_data.set_index('Chapter')