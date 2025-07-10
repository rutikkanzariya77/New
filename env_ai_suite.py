import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import re
import random
from datetime import datetime
import base64
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
import json

# Configure page
st.set_page_config(
    page_title="GreenAI: Smart Environmental AI Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E7D32, #4CAF50);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4CAF50;
        margin-top: 1rem;
    }
    .entity-tag {
        background: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        display: inline-block;
    }
    .stTab > div:first-child > div:first-child {
        gap: 0.5rem;
    }
    .stTab [data-baseweb="tab"] {
        background: #f0f0f0;
        border-radius: 10px 10px 0 0;
    }
    .stTab [data-baseweb="tab"]:hover {
        background: #e0e0e0;
    }
    .stTab [aria-selected="true"] {
        background: #4CAF50 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div class="main-header">
    <h1>🌱 GreenAI: Smart Environmental AI Assistant</h1>
    <p>Detects pollution types, generates visuals, fills blanks, and maps entities.</p>
</div>
""", unsafe_allow_html=True)

# Environmental categories for classification
ENVIRONMENTAL_CATEGORIES = {
    "Air Pollution": ["air pollution", "smog", "emissions", "carbon dioxide", "greenhouse gases", "particulate matter", "ozone"],
    "Water Pollution": ["water pollution", "contamination", "sewage", "industrial waste", "oil spill", "chemical runoff"],
    "Soil Contamination": ["soil pollution", "toxic waste", "pesticides", "heavy metals", "landfill", "contaminated land"],
    "Noise Pollution": ["noise pollution", "sound pollution", "traffic noise", "industrial noise", "urban noise"],
    "Climate Change": ["climate change", "global warming", "carbon footprint", "renewable energy", "deforestation", "melting ice"]
}

# Sample environmental entities for NER
ENVIRONMENTAL_ENTITIES = {
    "LOCATION": ["Amazon", "Pacific Ocean", "Sahara Desert", "Himalayas", "Great Barrier Reef", "Yellowstone", "Ganga", "Kanpur"],
    "POLLUTANT": ["CO2", "methane", "plastic", "mercury", "lead", "pesticides", "industrial waste", "toxic chemicals"],
    "ORGANIZATION": ["EPA", "Greenpeace", "WWF", "UNEP", "NASA", "NOAA"],
    "ENVIRONMENTAL": ["biodiversity", "ecosystem", "carbon footprint", "renewable energy", "sustainability", "conservation"]
}

# Fill-in-the-blank templates
FILL_BLANK_TEMPLATES = [
    "Air pollution is harmful as we <mask>.",
    "The <mask> river in Kanpur is polluted by plastic and industrial waste from textile factories.",
    "Deforestation leads to loss of <mask> and contributes to climate change.",
    "Solar energy is a <mask> source of power that doesn't produce emissions.",
    "Plastic waste in oceans harms <mask> life and disrupts food chains.",
    "Carbon <mask> from vehicles contribute to air pollution in urban areas.",
    "Recycling helps reduce <mask> and conserves natural resources.",
    "The <mask> layer protects Earth from harmful ultraviolet radiation."
]

# Functions for each feature
def classify_environmental_sentence(sentence):
    """Classify sentence into environmental categories"""
    sentence_lower = sentence.lower()
    scores = {}
    
    for category, keywords in ENVIRONMENTAL_CATEGORIES.items():
        score = sum(1 for keyword in keywords if keyword in sentence_lower)
        if score > 0:
            scores[category] = score
    
    if not scores:
        return "General Environmental", 0.5
    
    max_category = max(scores, key=scores.get)
    confidence = min(scores[max_category] / 3, 1.0)  # Normalize confidence
    
    return max_category, confidence

def extract_entities(text):
    """Extract named entities from text"""
    entities = []
    text_lower = text.lower()
    
    for entity_type, entity_list in ENVIRONMENTAL_ENTITIES.items():
        for entity in entity_list:
            if entity.lower() in text_lower:
                # Find the position in original text (case-sensitive)
                start_pos = text.lower().find(entity.lower())
                if start_pos != -1:
                    original_entity = text[start_pos:start_pos + len(entity)]
                    entities.append({
                        "text": original_entity,
                        "label": entity_type,
                        "start": start_pos,
                        "end": start_pos + len(entity)
                    })
    
    return entities

def create_entity_graph(entities):
    """Create a network graph from entities"""
    G = nx.Graph()
    
    # Add nodes
    for entity in entities:
        G.add_node(entity["text"], type=entity["label"])
    
    # Add edges between entities (simplified relationship)
    entity_texts = [e["text"] for e in entities]
    for i in range(len(entity_texts)):
        for j in range(i + 1, len(entity_texts)):
            G.add_edge(entity_texts[i], entity_texts[j])
    
    return G

def generate_environmental_image(prompt):
    """Generate a simple environmental image representation"""
    # Create a simple image with PIL (since we can't use actual AI image generation)
    img = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Simple visualization based on prompt
    if 'forest' in prompt.lower() or 'fire' in prompt.lower():
        # Draw trees and fire
        for i in range(5):
            x = 50 + i * 100
            # Tree trunk
            draw.rectangle([x-10, 350, x+10, 450], fill='brown')
            # Tree top
            draw.ellipse([x-30, 250, x+30, 350], fill='green')
            # Add fire effect
            if 'fire' in prompt.lower():
                draw.ellipse([x-25, 300, x+25, 380], fill='orange')
                draw.ellipse([x-15, 320, x+15, 360], fill='red')
    
    elif 'pollution' in prompt.lower():
        # Draw factory with smoke
        draw.rectangle([200, 300, 350, 450], fill='gray')
        draw.rectangle([250, 250, 270, 300], fill='darkgray')
        draw.rectangle([290, 250, 310, 300], fill='darkgray')
        # Smoke clouds
        for i in range(3):
            y = 150 + i * 30
            draw.ellipse([240, y, 280, y+40], fill='lightgray')
            draw.ellipse([280, y, 320, y+40], fill='lightgray')
    
    elif 'ocean' in prompt.lower() or 'water' in prompt.lower():
        # Draw ocean with waves
        for i in range(0, 512, 20):
            draw.arc([i-10, 250, i+30, 290], 0, 180, fill='blue', width=3)
        # Add some fish
        for i in range(3):
            x = 100 + i * 150
            draw.ellipse([x, 320, x+40, 340], fill='yellow')
    
    else:
        # Default: draw a landscape
        draw.rectangle([0, 350, 512, 512], fill='green')  # Ground
        draw.ellipse([100, 100, 200, 200], fill='yellow')  # Sun
        draw.rectangle([0, 0, 512, 350], fill='lightblue')  # Sky
    
    return img

def fill_mask_prediction(sentence, mask_token="<mask>"):
    """Predict fill-in-the-blank for environmental sentences"""
    sentence_lower = sentence.lower()
    
    # Simple rule-based predictions
    predictions = []
    
    if "pollution is harmful as we" in sentence_lower:
        predictions = ["breathe", "live", "survive", "exist", "inhale"]
    elif "river" in sentence_lower and "polluted" in sentence_lower:
        predictions = ["Ganga", "sacred", "holy", "major", "important"]
    elif "deforestation leads to loss of" in sentence_lower:
        predictions = ["biodiversity", "habitat", "wildlife", "trees", "forests"]
    elif "solar energy is a" in sentence_lower:
        predictions = ["renewable", "clean", "sustainable", "natural", "green"]
    elif "plastic waste" in sentence_lower and "harms" in sentence_lower:
        predictions = ["marine", "aquatic", "sea", "ocean", "underwater"]
    elif "carbon" in sentence_lower and "vehicles" in sentence_lower:
        predictions = ["emissions", "dioxide", "monoxide", "pollution", "gases"]
    elif "recycling helps reduce" in sentence_lower:
        predictions = ["waste", "pollution", "landfill", "garbage", "trash"]
    elif "layer protects earth" in sentence_lower:
        predictions = ["ozone", "atmospheric", "protective", "stratospheric", "natural"]
    else:
        predictions = ["environment", "pollution", "emissions", "waste", "conservation"]
    
    return predictions

# Main app interface
def main():
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Sentence Classification", "🖼️ Image Generation", "🔗 NER + Entity Graph", "📝 Fill-in-the-Blank"])
    
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🏷️ Environmental Sentence Classification")
        st.write("Classify environmental sentences into categories like Air Pollution, Water Pollution, etc.")
        
        # Input for sentence classification
        sentence_input = st.text_area(
            "Enter your environmental sentence:",
            placeholder="e.g., Air pollution from vehicles causes respiratory problems in urban areas.",
            height=100
        )
        
        if st.button("🔍 Classify Sentence", key="classify_btn"):
            if sentence_input:
                category, confidence = classify_environmental_sentence(sentence_input)
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.write(f"**Predicted Category:** {category}")
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Show all categories with scores
                st.subheader("📊 Category Scores:")
                scores_data = []
                for cat, keywords in ENVIRONMENTAL_CATEGORIES.items():
                    score = sum(1 for keyword in keywords if keyword in sentence_input.lower())
                    scores_data.append({"Category": cat, "Score": score})
                
                df_scores = pd.DataFrame(scores_data)
                fig = px.bar(df_scores, x="Category", y="Score", 
                           title="Environmental Category Scores",
                           color="Score", color_continuous_scale="Greens")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🎨 Environmental Image Generation")
        st.write("Generate environmental images based on text prompts.")
        
        # Input for image generation
        image_prompt = st.text_input(
            "Enter environmental image prompt:",
            placeholder="e.g., Forest fire, Ocean pollution, Solar farm"
        )
        
        if st.button("🖼️ Generate Image", key="image_btn"):
            if image_prompt:
                with st.spinner("Generating environmental image..."):
                    generated_image = generate_environmental_image(image_prompt)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(generated_image, caption=f"Generated: {image_prompt}", use_column_width=True)
                    
                    # Show image analysis
                    st.subheader("📋 Image Analysis")
                    analysis = f"Generated image represents: {image_prompt}"
                    if 'fire' in image_prompt.lower():
                        analysis += "\n- Shows environmental hazard (wildfire)"
                        analysis += "\n- Depicts impact on forest ecosystem"
                    elif 'pollution' in image_prompt.lower():
                        analysis += "\n- Shows industrial pollution source"
                        analysis += "\n- Depicts air quality impact"
                    elif 'ocean' in image_prompt.lower():
                        analysis += "\n- Shows marine environment"
                        analysis += "\n- Depicts water ecosystem"
                    
                    st.text_area("Analysis:", value=analysis, height=100, disabled=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🔗 Named Entity Recognition + Entity Graph")
        st.write("Extract environmental entities and visualize their relationships.")
        
        # Input for NER
        ner_input = st.text_area(
            "Enter sentence with named entities:",
            placeholder="e.g., The Ganga river in Kanpur is polluted by plastic and industrial waste from textile factories.",
            height=100
        )
        
        if st.button("🔍 Extract Entities", key="ner_btn"):
            if ner_input:
                entities = extract_entities(ner_input)
                
                if entities:
                    st.subheader("🏷️ Detected Entities:")
                    
                    # Display entities as tags
                    entity_html = ""
                    for entity in entities:
                        entity_html += f'<span class="entity-tag">{entity["text"]} ({entity["label"]})</span> '
                    
                    st.markdown(entity_html, unsafe_allow_html=True)
                    
                    # Create and display entity graph
                    st.subheader("📊 Entity Relationship Graph:")
                    
                    G = create_entity_graph(entities)
                    
                    # Create plotly network graph
                    pos = nx.spring_layout(G, k=3, iterations=50)
                    
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                         line=dict(width=2, color='#888'),
                                         hoverinfo='none',
                                         mode='lines')
                    
                    node_x = []
                    node_y = []
                    node_text = []
                    node_colors = []
                    
                    color_map = {
                        'LOCATION': '#FF6B6B',
                        'POLLUTANT': '#4ECDC4',
                        'ORGANIZATION': '#45B7D1',
                        'ENVIRONMENTAL': '#96CEB4'
                    }
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node)
                        
                        # Get node type
                        node_type = next((e["label"] for e in entities if e["text"] == node), "UNKNOWN")
                        node_colors.append(color_map.get(node_type, '#888'))
                    
                    node_trace = go.Scatter(x=node_x, y=node_y,
                                          mode='markers+text',
                                          hoverinfo='text',
                                          text=node_text,
                                          textposition="middle center",
                                          marker=dict(size=20,
                                                    color=node_colors,
                                                    line=dict(width=2, color='white')))
                    
                    fig = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      title='Environmental Entity Relationship Graph',
                                      titlefont_size=16,
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20,l=5,r=5,t=40),
                                      annotations=[ dict(
                                          text="Entities and their relationships",
                                          showarrow=False,
                                          xref="paper", yref="paper",
                                          x=0.005, y=-0.002 ) ],
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Entity summary
                    st.subheader("📋 Entity Summary:")
                    entity_df = pd.DataFrame(entities)
                    st.dataframe(entity_df[['text', 'label']], use_container_width=True)
                    
                else:
                    st.warning("No environmental entities found. Try sentences with locations, pollutants, or organizations.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📝 Fill-in-the-Blank (Environmental Mask)")
        st.write("Complete environmental sentences using AI predictions.")
        
        # Dropdown for template selection
        template_choice = st.selectbox(
            "Choose a template or enter your own:",
            ["Custom"] + FILL_BLANK_TEMPLATES
        )
        
        if template_choice == "Custom":
            mask_input = st.text_input(
                "Enter sentence with <mask>:",
                placeholder="e.g., Renewable energy sources like solar and wind are <mask> for the environment."
            )
        else:
            mask_input = template_choice
            st.text_area("Selected template:", value=mask_input, height=80, disabled=True)
        
        if st.button("🎯 Predict Fill-in-the-Blank", key="mask_btn"):
            if mask_input and "<mask>" in mask_input:
                predictions = fill_mask_prediction(mask_input)
                
                st.subheader("🎯 Top Predictions:")
                
                # Display predictions with confidence scores
                for i, prediction in enumerate(predictions[:5], 1):
                    confidence = max(0.95 - (i-1) * 0.1, 0.5)  # Simulated confidence
                    filled_sentence = mask_input.replace("<mask>", f"**{prediction}**")
                    
                    st.markdown(f"**{i}.** {filled_sentence}")
                    st.caption(f"Confidence: {confidence:.1%}")
                    st.markdown("---")
                
                # Show all predictions in a chart
                pred_df = pd.DataFrame({
                    'Prediction': predictions[:5],
                    'Confidence': [max(0.95 - i * 0.1, 0.5) for i in range(5)]
                })
                
                fig = px.bar(pred_df, x='Prediction', y='Confidence',
                           title='Prediction Confidence Scores',
                           color='Confidence', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("Please enter a sentence with <mask> token.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar with additional info
    with st.sidebar:
        st.markdown("### 🌱 GreenAI Features")
        st.markdown("""
        - **Sentence Classification**: Categorize environmental statements
        - **Image Generation**: Create environmental visualizations
        - **NER + Entity Graph**: Extract and map environmental entities
        - **Fill-in-the-Blank**: Complete environmental sentences
        """)
        
        st.markdown("### 📊 Statistics")
        st.metric("Environmental Categories", len(ENVIRONMENTAL_CATEGORIES))
        st.metric("Entity Types", len(ENVIRONMENTAL_ENTITIES))
        st.metric("Fill-in Templates", len(FILL_BLANK_TEMPLATES))
        
        st.markdown("### 🔧 About")
        st.markdown("""
        This AI assistant helps with environmental text analysis and content generation. 
        Built with Streamlit and various ML techniques for environmental domain applications.
        """)

if __name__ == "__main__":
    main()
