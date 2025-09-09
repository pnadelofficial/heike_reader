import streamlit as st
from utils import load_data, display_sentence, Searcher

st.title("Heike Reader")

glosses, translated = load_data()

chapter_select = st.sidebar.selectbox(
    "Select Chapter",
    options=translated['chapter_id'].unique(),
    index=0,
    format_func=lambda x: f"Chapter {x}"
)

st.sidebar.markdown("---")

search_bar = st.sidebar.text_input("Search for a word to see its occurrences")

if search_bar:
    searcher = Searcher(translated, glosses)
    results = searcher.search(search_bar)

else:
    translated_subset = translated[translated['chapter_id'] == chapter_select].reset_index(drop=True)
    st.write(f"Number of sentences in Chapter {chapter_select}: {len(translated_subset)}")

    for i, row in translated_subset.iterrows():
        sentence = row.original
        translation = row.translation
        st.markdown(f"##### Sentence {i+1}")
        display_sentence(sentence, translation, glosses)
        st.markdown("---")
