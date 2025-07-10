import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Load resources
classifier = pipeline("zero-shot-classification")
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
nlp = spacy.load("en_core_web_sm")

# Set UI
st.set_page_config(page_title="GreenAI 🌍", layout="wide")
st.title("🌱 GreenAI: Smart Environmental AI Assistant")

tabs = st.tabs(["1️⃣ Sentence Classification", "2️⃣ Image Generation", "3️⃣ NER + Entity Graph", "4️⃣ Fill-in-the-Blank"])

# ----------------------------------------
# 1. Sentence Classification
# ----------------------------------------
with tabs[0]:
    st.header("1️⃣ Sentence Classification")
    user_input = st.text_area("Enter an environmental sentence to classify:", "")
    labels = ["Pollution", "Climate Change", "Wildlife", "Water Resources", "Waste Management"]

    if st.button("Classify"):
        if user_input:
            result = classifier(user_input, labels)
            st.subheader("Classification Result:")
            for label, score in zip(result["labels"], result["scores"]):
                st.write(f"**{label}**: {score:.2f}")
        else:
            st.warning("Please enter a sentence.")

# ----------------------------------------
# 2. Image Generation
# ----------------------------------------
with tabs[1]:
    st.header("2️⃣ Image Generation")
    prompt = st.text_input("Enter an environmental image prompt (e.g. Forest fire):")

    if st.button("Generate Image"):
        if prompt:
            from diffusers import StableDiffusionPipeline
            import torch

            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            with st.spinner("Generating image..."):
                image = pipe(prompt).images[0]
                st.image(image, caption=f"Generated for: {prompt}", use_column_width=True)
        else:
            st.warning("Please enter a prompt.")

# ----------------------------------------
# 3. NER + Entity Graph
# ----------------------------------------
with tabs[2]:
    st.header("3️⃣ NER + Entity Relationship Graph")
    ner_input = st.text_area("Enter a sentence with environmental entities:", "The Ganga river in Kanpur is polluted by plastic and industrial waste.")

    if st.button("Extract Entities & Generate Graph"):
        doc = nlp(ner_input)
        st.subheader("Entities Found:")
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        for text, label in ents:
            st.write(f"• **{text}** → *{label}*")

        # Draw Graph
        G = nx.Graph()
        for ent in ents:
            G.add_node(ent[0], label=ent[1])

        for i in range(len(ents)-1):
            G.add_edge(ents[i][0], ents[i+1][0])

        fig, ax = plt.subplots()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, edge_color='gray')
        st.pyplot(fig)

# ----------------------------------------
# 4. Fill-in-the-Blank
# ----------------------------------------
with tabs[3]:
    st.header("4️⃣ Fill-in-the-Blank with Mask")
    mask_input = st.text_input("Enter a sentence with `<mask>`:", "<mask> pollution is harmful as we breathe.")

    if st.button("Predict Mask"):
        if "<mask>" not in mask_input:
            st.warning("Please include the token '<mask>' in your sentence.")
        else:
            result = fill_mask(mask_input)
            st.subheader("Top Predictions:")
            for r in result:
                st.write(f"• {r['sequence']} (score: {r['score']:.4f})")
