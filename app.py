import streamlit as st
from utils import load_data, display_sentence, Searcher

st.title("Heike Reader")

if 'show_info' not in st.session_state:
    st.session_state.show_info = False

with st.expander("ℹ️ How to Use Heike Reader"):
    st.markdown("""
    **📖 Reading Mode**
    - Select any chapter from the sidebar
    - Click on Japanese tokens to see detailed annotations (transliteration, lemma, part of speech, gloss)
    - Annotations appear below each sentence and can be copied
    
    **🔍 Search Features**
    - Search for words in Japanese characters, romaji transliteration, or lemmas
    - Supports partial matching (e.g., "ky" finds "kyou", "kyoku", etc.)
    - Results show match types and highlight found terms in context
    - Search across all chapters simultaneously
    
    **💡 Tips**
    - Try searching in different ways: Japanese (今日), romaji (kyou), or partial terms
    - Click multiple tokens in a sentence to compare their annotations
    - Use the search to find patterns across the entire text
    """)

glosses, translated = load_data()

chapter_select = st.sidebar.selectbox(
    "Select Chapter",
    options=translated['chapter_id'].unique(),
    index=0,
    format_func=lambda x: f"Chapter {x}"
)

st.sidebar.markdown("---")

search_bar = st.sidebar.text_input("Search for a word to see its occurrences", key="search_input")

if search_bar:
    searcher = Searcher(translated, glosses)
    results = searcher.search(search_bar.strip())

else:
    translated_subset = translated[translated['chapter_id'] == chapter_select].reset_index(drop=True)
    st.write(f"Number of sentences in Chapter {chapter_select}: {len(translated_subset)}")

    for i, row in translated_subset.iterrows():
        sentence = row.original
        translation = row.translation
        st.markdown(f"##### Sentence {i+1}")
        display_sentence(sentence, translation, glosses, sent_id=i)
