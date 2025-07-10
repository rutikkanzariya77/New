import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import spacy
import re
from PIL import Image, ImageDraw, ImageFont
import random
import io
import base64

# Set page config
st.set_page_config(
    page_title="Environmental AI Suite",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .section-header {
        font-size: 2rem;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .stApp {
        background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
    }
    .stApp > div {
        background: transparent;
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stSelectbox label, .stTextArea label, .stTextInput label, .stSlider label {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
    .stSidebar {
        background: linear-gradient(180deg, #2c5530 0%, #1a3d1f 100%);
    }
    .stSidebar .stSelectbox label {
        color: #FFFFFF !important;
    }
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classification_model' not in st.session_state:
    st.session_state.classification_model = None
if 'ner_model' not in st.session_state:
    st.session_state.ner_model = None

class EnvironmentalClassifier:
    def __init__(self):
        self.categories = {
            0: "Climate Change",
            1: "Pollution Control",
            2: "Wildlife Conservation",
            3: "Renewable Energy",
            4: "Waste Management"
        }
        
        # Training data
        self.training_data = [
            ("Global warming is causing ice caps to melt rapidly", 0),
            ("Carbon emissions need to be reduced immediately", 0),
            ("The greenhouse effect is intensifying", 0),
            ("Industrial waste is contaminating water sources", 1),
            ("Air pollution levels are reaching dangerous heights", 1),
            ("Chemical runoff is destroying marine ecosystems", 1),
            ("Endangered species need protection from habitat loss", 2),
            ("Wildlife corridors help animals migrate safely", 2),
            ("Deforestation is threatening biodiversity", 2),
            ("Solar panels are becoming more efficient", 3),
            ("Wind energy is a clean alternative to fossil fuels", 3),
            ("Hydroelectric power generates renewable electricity", 3),
            ("Recycling programs reduce landfill waste", 4),
            ("Composting organic matter enriches soil naturally", 4),
            ("Plastic waste is accumulating in oceans", 4),
        ]
        
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        texts = [item[0] for item in self.training_data]
        labels = [item[1] for item in self.training_data]
        self.model.fit(texts, labels)
    
    def predict(self, text):
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        return self.categories[prediction], probabilities

class ImageGenerator:
    def __init__(self):
        self.themes = {
            "Forest": {"colors": ["#228B22", "#32CD32", "#006400"], "elements": ["🌲", "🌳", "🍃"]},
            "Ocean": {"colors": ["#4682B4", "#87CEEB", "#000080"], "elements": ["🌊", "🐠", "🐙"]},
            "Desert": {"colors": ["#F4A460", "#DEB887", "#CD853F"], "elements": ["🌵", "🐪", "☀️"]},
            "Mountain": {"colors": ["#708090", "#2F4F4F", "#FFFFFF"], "elements": ["⛰️", "🏔️", "🦅"]},
            "Urban": {"colors": ["#696969", "#A9A9A9", "#32CD32"], "elements": ["🏢", "🌱", "♻️"]}
        }
    
    def generate_environmental_image(self, theme, width=800, height=600):
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        theme_data = self.themes.get(theme, self.themes["Forest"])
        colors = theme_data["colors"]
        elements = theme_data["elements"]
        
        # Create gradient background
        for i in range(height):
            color_ratio = i / height
            r = int(colors[0][1:3], 16) * (1 - color_ratio) + int(colors[1][1:3], 16) * color_ratio
            g = int(colors[0][3:5], 16) * (1 - color_ratio) + int(colors[1][3:5], 16) * color_ratio
            b = int(colors[0][5:7], 16) * (1 - color_ratio) + int(colors[1][5:7], 16) * color_ratio
            draw.rectangle([0, i, width, i+1], fill=(int(r), int(g), int(b)))
        
        # Add shapes representing environmental elements
        for _ in range(random.randint(15, 25)):
            x = random.randint(0, width-100)
            y = random.randint(0, height-100)
            size = random.randint(30, 80)
            color = random.choice(colors)
            
            # Convert hex to RGB
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            if theme == "Forest":
                draw.ellipse([x, y, x+size, y+size], fill=color_rgb)
            elif theme == "Ocean":
                draw.polygon([(x, y+size), (x+size//2, y), (x+size, y+size)], fill=color_rgb)
            elif theme == "Desert":
                draw.rectangle([x, y, x+size, y+size], fill=color_rgb)
            elif theme == "Mountain":
                draw.polygon([(x, y+size), (x+size//2, y), (x+size, y+size)], fill=color_rgb)
            else:  # Urban
                draw.rectangle([x, y, x+size, y+size//2], fill=color_rgb)
        
        return img

class NERProcessor:
    def __init__(self):
        # Environmental entities patterns
        self.environmental_patterns = {
            "SPECIES": ["tiger", "elephant", "whale", "eagle", "bear", "wolf", "dolphin", "shark"],
            "ECOSYSTEM": ["forest", "ocean", "desert", "grassland", "wetland", "coral reef", "rainforest"],
            "POLLUTANT": ["carbon dioxide", "methane", "plastic", "oil", "chemical", "pesticide", "mercury"],
            "RESOURCE": ["water", "air", "soil", "mineral", "fossil fuel", "renewable energy", "solar"],
            "LOCATION": ["Amazon", "Arctic", "Pacific", "Atlantic", "Sahara", "Antarctica", "Himalaya"]
        }
    
    def extract_entities(self, text):
        entities = []
        text_lower = text.lower()
        
        for entity_type, keywords in self.environmental_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    start = text_lower.find(keyword)
                    end = start + len(keyword)
                    entities.append({
                        "text": text[start:end],
                        "label": entity_type,
                        "start": start,
                        "end": end
                    })
        
        return entities
    
    def create_knowledge_graph(self, entities):
        G = nx.Graph()
        
        # Add nodes
        for entity in entities:
            G.add_node(entity["text"], type=entity["label"])
        
        # Add edges (simple co-occurrence based)
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1["label"] != entity2["label"]:
                    G.add_edge(entity1["text"], entity2["text"])
        
        return G

class MaskFiller:
    def __init__(self):
        self.environmental_contexts = {
            "climate": [
                "The [MASK] is causing global temperatures to rise.",
                "Reducing [MASK] emissions is crucial for climate stability.",
                "The [MASK] effect traps heat in Earth's atmosphere."
            ],
            "pollution": [
                "Industrial [MASK] contaminates water sources.",
                "Air [MASK] affects human health and environment.",
                "Plastic [MASK] accumulates in ocean ecosystems."
            ],
            "conservation": [
                "Wildlife [MASK] protects endangered species.",
                "Forest [MASK] prevents habitat destruction.",
                "Marine [MASK] areas safeguard ocean biodiversity."
            ],
            "energy": [
                "Solar [MASK] converts sunlight into electricity.",
                "Wind [MASK] generate clean renewable power.",
                "Hydroelectric [MASK] harness water flow for energy."
            ],
            "waste": [
                "Recycling [MASK] reduces landfill accumulation.",
                "Composting [MASK] enriches soil naturally.",
                "Waste [MASK] programs minimize environmental impact."
            ]
        }
        
        self.answers = {
            "greenhouse": ["greenhouse effect", "greenhouse gases"],
            "carbon": ["carbon dioxide", "carbon emissions"],
            "waste": ["waste materials", "waste products"],
            "pollution": ["pollution levels", "pollution sources"],
            "conservation": ["conservation efforts", "conservation programs"],
            "panels": ["solar panels", "photovoltaic panels"],
            "turbines": ["wind turbines", "wind generators"],
            "plants": ["power plants", "hydroelectric plants"],
            "programs": ["recycling programs", "waste programs"],
            "organic": ["organic waste", "organic matter"],
            "management": ["waste management", "resource management"]
        }
    
    def generate_masked_sentence(self, category=None):
        if category:
            sentences = self.environmental_contexts.get(category, [])
            if sentences:
                return random.choice(sentences)
        
        # Random selection from all categories
        all_sentences = []
        for sentences in self.environmental_contexts.values():
            all_sentences.extend(sentences)
        return random.choice(all_sentences)
    
    def get_suggestions(self, masked_sentence):
        suggestions = []
        for key, values in self.answers.items():
            for value in values:
                if any(word in masked_sentence.lower() for word in value.split()):
                    suggestions.extend(values)
                    break
        return list(set(suggestions))[:5]

# Initialize components
@st.cache_resource
def load_models():
    classifier = EnvironmentalClassifier()
    image_gen = ImageGenerator()
    ner_processor = NERProcessor()
    mask_filler = MaskFiller()
    return classifier, image_gen, ner_processor, mask_filler

def main():
    st.markdown('<h1 class="main-header">🌱 Environmental AI Suite</h1>', unsafe_allow_html=True)
    
    # Load models
    classifier, image_gen, ner_processor, mask_filler = load_models()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a module:",
        ["🏠 Home", "📊 Sentence Classification", "🎨 Image Generation", "🔍 NER & Graph Mapping", "📝 Fill in the Blanks"]
    )
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Environmental AI Suite
        
        This application provides four powerful AI tools for environmental analysis:
        
        ### 🎯 Sentence Classification
        Classify environmental sentences into 5 categories:
        - Climate Change
        - Pollution Control
        - Wildlife Conservation
        - Renewable Energy
        - Waste Management
        
        ### 🎨 Image Generation
        Generate environmental-themed images with various themes:
        - Forest ecosystems
        - Ocean environments
        - Desert landscapes
        - Mountain regions
        - Urban sustainability
        
        ### 🔍 Named Entity Recognition & Graph Mapping
        Extract environmental entities and visualize their relationships:
        - Species identification
        - Ecosystem mapping
        - Pollutant detection
        - Resource analysis
        - Location recognition
        
        ### 📝 Fill in the Blanks
        Interactive environmental knowledge testing with masked language modeling for various environmental topics.
        
        **Select a module from the sidebar to get started!**
        """)
    
    elif page == "📊 Sentence Classification":
        st.markdown('<h2 class="section-header">Sentence Classification</h2>', unsafe_allow_html=True)
        
        st.write("Classify environmental sentences into different categories:")
        
        # Text input
        user_text = st.text_area("Enter an environmental sentence:", 
                                placeholder="e.g., Solar panels are becoming more efficient and affordable")
        
        if st.button("Classify Sentence"):
            if user_text.strip():
                category, probabilities = classifier.predict(user_text)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Predicted Category:** {category}")
                    st.write(f"**Confidence:** {max(probabilities):.2%}")
                
                with col2:
                    # Create probability chart
                    fig = px.bar(
                        x=list(classifier.categories.values()),
                        y=probabilities,
                        title="Classification Probabilities",
                        labels={"x": "Categories", "y": "Probability"}
                    )
                    st.plotly_chart(fig)
        
        # Example sentences
        st.subheader("Try these examples:")
        examples = [
            "Melting glaciers are contributing to sea level rise",
            "Factory emissions are polluting the atmosphere",
            "Pandas are endangered due to habitat loss",
            "Wind farms generate clean electricity",
            "Recycling plastic bottles reduces waste"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}: {example}", key=f"example_{i}"):
                category, probabilities = classifier.predict(example)
                st.info(f"Category: **{category}** (Confidence: {max(probabilities):.2%})")
    
    elif page == "🎨 Image Generation":
        st.markdown('<h2 class="section-header">Environmental Image Generation</h2>', unsafe_allow_html=True)
        
        st.write("Generate environmental-themed images based on different ecosystems:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Select Theme:", list(image_gen.themes.keys()))
            width = st.slider("Width", 400, 1200, 800)
            height = st.slider("Height", 300, 900, 600)
            
            if st.button("Generate Image"):
                with st.spinner("Generating environmental image..."):
                    img = image_gen.generate_environmental_image(theme, width, height)
                    st.image(img, caption=f"Generated {theme} Environment")
                    
                    # Convert to bytes for download
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    st.download_button(
                        label="Download Image",
                        data=img_bytes,
                        file_name=f"environmental_{theme.lower()}_{width}x{height}.png",
                        mime="image/png"
                    )
        
        with col2:
            st.subheader("Theme Characteristics:")
            selected_theme = image_gen.themes[theme]
            st.write(f"**Colors:** {', '.join(selected_theme['colors'])}")
            st.write(f"**Elements:** {', '.join(selected_theme['elements'])}")
            
            # Show color palette
            fig, ax = plt.subplots(1, 1, figsize=(8, 2))
            colors = selected_theme['colors']
            for i, color in enumerate(colors):
                ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
            ax.set_xlim(0, len(colors))
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"{theme} Color Palette")
            st.pyplot(fig)
    
    elif page == "🔍 NER & Graph Mapping":
        st.markdown('<h2 class="section-header">Named Entity Recognition & Graph Mapping</h2>', unsafe_allow_html=True)
        
        st.write("Extract environmental entities and visualize their relationships:")
        
        # Text input
        text_input = st.text_area("Enter environmental text:", 
                                 placeholder="e.g., The Amazon rainforest is home to jaguars and provides oxygen while facing deforestation threats from oil drilling.",
                                 height=100)
        
        if st.button("Extract Entities & Create Graph"):
            if text_input.strip():
                entities = ner_processor.extract_entities(text_input)
                
                if entities:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Extracted Entities:")
                        entity_df = pd.DataFrame(entities)
                        st.dataframe(entity_df[['text', 'label']])
                        
                        # Entity statistics
                        entity_counts = Counter([e['label'] for e in entities])
                        fig = px.pie(
                            values=list(entity_counts.values()),
                            names=list(entity_counts.keys()),
                            title="Entity Distribution"
                        )
                        st.plotly_chart(fig)
                    
                    with col2:
                        st.subheader("Knowledge Graph:")
                        G = ner_processor.create_knowledge_graph(entities)
                        
                        if G.nodes():
                            # Create network visualization
                            pos = nx.spring_layout(G, k=1, iterations=50)
                            
                            # Extract node and edge information
                            node_trace = go.Scatter(
                                x=[pos[node][0] for node in G.nodes()],
                                y=[pos[node][1] for node in G.nodes()],
                                mode='markers+text',
                                text=[node for node in G.nodes()],
                                textposition="middle center",
                                marker=dict(size=20, color='lightblue'),
                                name="Entities"
                            )
                            
                            edge_trace = []
                            for edge in G.edges():
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                edge_trace.append(go.Scatter(
                                    x=[x0, x1, None],
                                    y=[y0, y1, None],
                                    mode='lines',
                                    line=dict(width=2, color='gray'),
                                    showlegend=False
                                ))
                            
                            fig = go.Figure(data=[node_trace] + edge_trace)
                            fig.update_layout(
                                title="Environmental Entity Network",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                annotations=[ dict(
                                    text="Entities and their relationships",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002,
                                    xanchor="left", yanchor="bottom",
                                    font=dict(size=12)
                                )],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            )
                            st.plotly_chart(fig)
                        else:
                            st.info("No relationships found between entities.")
                else:
                    st.warning("No environmental entities found in the text.")
        
        # Example texts
        st.subheader("Try these examples:")
        examples = [
            "The Amazon rainforest is home to jaguars and provides oxygen while facing deforestation threats.",
            "Arctic ice melting affects polar bears and releases methane into the atmosphere.",
            "Solar panels in the Sahara could provide renewable energy while reducing carbon emissions.",
            "Pacific Ocean plastic pollution threatens whales and damages coral reef ecosystems."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"ner_example_{i}"):
                entities = ner_processor.extract_entities(example)
                if entities:
                    st.json([{"text": e["text"], "label": e["label"]} for e in entities])
    
    elif page == "📝 Fill in the Blanks":
        st.markdown('<h2 class="section-header">Environmental Fill in the Blanks</h2>', unsafe_allow_html=True)
        
        st.write("Test your environmental knowledge with fill-in-the-blank questions:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox("Select Category:", 
                                   ["Random"] + list(mask_filler.environmental_contexts.keys()))
            
            if st.button("Generate New Question"):
                if category == "Random":
                    sentence = mask_filler.generate_masked_sentence()
                else:
                    sentence = mask_filler.generate_masked_sentence(category)
                
                st.session_state.current_sentence = sentence
                st.session_state.user_answer = ""
            
            # Display current question
            if hasattr(st.session_state, 'current_sentence'):
                st.subheader("Fill in the blank:")
                st.write(st.session_state.current_sentence)
                
                user_answer = st.text_input("Your answer:", key="fill_answer")
                
                if st.button("Check Answer"):
                    if user_answer.strip():
                        suggestions = mask_filler.get_suggestions(st.session_state.current_sentence)
                        
                        # Simple scoring
                        is_correct = any(user_answer.lower() in suggestion.lower() for suggestion in suggestions)
                        
                        if is_correct:
                            st.success("✅ Great answer!")
                        else:
                            st.error("❌ Try again!")
                            st.info(f"Hint: Consider words related to {suggestions[0] if suggestions else 'environmental topics'}")
        
        with col2:
            st.subheader("Categories Available:")
            for cat, sentences in mask_filler.environmental_contexts.items():
                st.write(f"**{cat.title()}:** {len(sentences)} questions")
            
            # Quick examples
            st.subheader("Example Questions:")
            for i, (cat, sentences) in enumerate(list(mask_filler.environmental_contexts.items())[:3]):
                st.write(f"**{cat.title()}:** {sentences[0]}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #FFFFFF; padding: 2rem 0; background: rgba(0,0,0,0.2); border-radius: 10px; margin-top: 2rem;'>
        <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>Environmental AI Suite - Powered by Streamlit</p>
        <p style='font-size: 1.1rem; margin: 0;'>🌍 Building a sustainable future through AI 🌱</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
