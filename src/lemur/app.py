import spacy_streamlit
import streamlit

# Based on https://github.com/explosion/spacy-streamlit/blob/master/examples/02_custom.py

# Note that you need to run `python -m spacy download en_core_sci_sm` first.
# spacy_model = "en_core_web_sm"
spacy_model = "en_core_sci_sm" # From https://github.com/monarch-initiative/monarch-app/blob/9358b8aa1f7549da087734276c98739d6624aa03/backend/src/monarch_py/implementations/spacy/spacy_implementation.py#L23C89-L23C103

# Set up streamlit
streamlit.title("LEMUR: Linking Entities Multipipeline Unified Ranking")
pubmed_url = streamlit.text_input("PubMed URL", "https://www.ncbi.nlm.nih.gov/pubmed/33999999")
text = streamlit.text_area("Text to analyze", "Enter text here.", height=200)
doc = spacy_streamlit.process_text(spacy_model, text)

spacy_streamlit.visualize_ner(
    doc,
    labels=[],
    show_table=True,
    title="Disease/phenotype, gene, protein",
)

nlp = spacy_streamlit.load_model("en_core_web_md")
spacy_streamlit.visualize_similarity(
    nlp,
    default_texts=("apple", "orange"),
    threshold=0.5,
    title="Compare Word Similarity"
)

streamlit.text(f"Analyzed using spaCy model {spacy_model}")
