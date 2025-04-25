import streamlit as st
import openai
import pandas as pd
import time
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
import faiss
import numpy as np
import os
import pickle
import json
from prompts import (
    get_meal_prompt, 
    SYSTEM_PROMPT,
    DIETARY_REQUIREMENTS,
    IMPORTANT_RULES,
    AUTHENTIC_RECIPE_NAMES
)

# Set page config
st.set_page_config(
    page_title="BiteWise",
    page_icon="🍽️",
    layout="wide"
)

from prompts import HEALTH_RESTRICTIONS, ALLERGEN_KEYWORDS, DIET_RESTRICTIONS

import random
import re
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RESET_FAISS_ON_START = True  # Toggle this for clean dev runs

# ========== FAISS Embedding Index Setup ==========
EMBEDDING_DIM = 1536
INDEX_FILE = "recipe_index.faiss"
NAMES_FILE = "recipe_names.pkl"

if RESET_FAISS_ON_START:
    recipe_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    recipe_names = []
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(NAMES_FILE):
        os.remove(NAMES_FILE)
else:
    if os.path.exists(INDEX_FILE):
        recipe_index = faiss.read_index(INDEX_FILE)
    else:
        recipe_index = faiss.IndexFlatL2(EMBEDDING_DIM)

    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "rb") as f:
            recipe_names = pickle.load(f)
    else:
        recipe_names = []

stored_embeddings = []  # (name, ingredients, embedding)

session = None

async def init_http_session():
    global session
    if session is None:
        session = aiohttp.ClientSession()

async def close_http_session():
    global session
    if session:
        await session.close()
        session = None
        
def parse_list(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                return eval(value)
            except:
                return [value]
        return [value]
    return value

# Common household measurements with focus on Australian staples
COMMON_MEASUREMENTS = {
    "can": ["chickpeas", "tomatoes", "coconut milk", "lentils", "beans", "tuna", "corn", "baked beans", "diced tomatoes"],
    "bunch": ["green onions", "parsley", "coriander", "mint", "rocket"],
    "head": ["garlic", "lettuce", "cabbage"],
    "whole": ["lemon", "lime", "orange", "potato", "onion", "carrot", "tomato", "zucchini", "capsicum", "mushroom"],
    "package": ["frozen vegetables", "frozen peas", "frozen corn", "mixed vegetables"],
    "kg": ["chicken thighs", "chicken breasts", "prawn", "beef mince", "pork shoulder", "potatoes", "rice", "onions", "carrots", "kale", "spinach", "mushroom"],
    "gram": ["cheese", "greek yogurt", "butter", "feta", "ricotta"],
    "slice": ["bread", "toast", "rye bread"],
    "dozen": ["eggs"],
    "unit": ["zucchini", "cucumber", "capsicum", "eggplant", "avocado", "mushroom", "bagel"]
}

CATEGORIES = {
    "Fresh & Frozen Produce": [
        "potato", "onion", "carrot", "cabbage", "pumpkin", "sweet potato",
        "zucchini", "cucumber", "capsicum", "eggplant", "broccoli", "kale",
        "spinach", "rocket", "mushroom", "tomato", "lemon", "lime", "avocado",
        "lettuce", "spring onion", "garlic", "ginger", "chilli", "radish", "daikon",
        "bok choy", "snow pea", "bean sprout", "turnip", "turnip leaves",
        "perilla leaves", "seaweed", "beetroot", "corn", "asparagus", "frozen peas",
        "frozen corn", "mixed vegetables", "frozen spinach", "frozen berries", "frozen edamame"
    ],
    "Budget Proteins": [
        "chicken thighs", "chicken breasts", "prawn", "beef mince", "pork shoulder", "whole chicken",
        "turkey mince", "salmon", "tuna", "sardine", "haddock", "egg", "tofu", "tempeh", "lentil",
        "chickpea", "black bean", "white bean", "kidney bean", "edamame", "anchovy", "pollock", "mackerel",
        "smoked tofu", "soft tofu", "firm tofu", "silken tofu", "canned tuna", "canned salmon"
    ],
    "Dairy & Eggs": [
        "greek yogurt", "cheese", "milk", "butter", "egg", "feta", "ricotta", "yogurt",
        "parmesan", "mozzarella", "cottage cheese", "plant-based milk", "plant-based yogurt"
    ],
    "Affordable Grains": [
        "rice", "brown rice", "wild rice", "basmati rice", "jasmine rice",
        "pasta", "spelt pasta", "barley", "bulgur", "millet", "oat", "rolled oats", "steel-cut oats",
        "quinoa", "couscous", "bread", "wholegrain bread", "rye bread", "sourdough", "bagel",
        "tortilla", "cornbread", "noodle", "soba noodle", "vermicelli", "udon", "ramen", "glass noodle"
    ],
    "Pantry Essentials": [
        "tomato paste", "canned tomato", "canned corn", "canned beans", "canned chickpeas",
        "baked beans", "stock cube", "vegetable stock", "chicken stock", "beef stock",
        "flour", "wholemeal flour", "cornstarch", "sugar", "brown sugar", "honey", "maple syrup",
        "mustard", "bbq sauce", "dijon mustard", "gochujang", "doenjang", "miso paste",
        "rice vinegar", "apple cider vinegar", "white vinegar", "coconut milk", "canned coconut milk"
    ],
    "Herbs & Spices": [
        "parsley", "coriander", "mint", "basil", "thyme", "paprika", "smoked paprika",
        "cumin", "oregano", "dill", "chives", "black pepper", "white pepper",
        "mustard powder", "turmeric", "fenugreek", "cinnamon", "clove", "nutmeg",
        "star anise", "bay leaf", "rosemary", "chilli flakes", "cardamom", "curry powder"
    ],
    "Sauces & Condiments": [
        "oil", "olive oil", "vegetable oil", "sesame oil", "soy sauce", "tamari", "fish sauce",
        "mayonnaise", "vegan mayo", "tomato sauce", "bbq sauce", "yogurt sauce", "lemon juice",
        "lime juice", "vinegar", "sambal oelek", "sriracha", "hoisin sauce", "tahini", "nut butter"
    ]
}

STRUCTURE_KEYWORDS = {
    "curry": ["curry", "korma", "masala", "tikka"],
    "bowl": ["bowl", "rice bowl", "nourish bowl", "grain bowl"],
    "wrap": ["wrap", "roll", "burrito", "fajita"],
    "stirfry": ["stir-fry", "stir fry", "sauté"],
    "salad": ["salad", "slaw", "coleslaw"],
    "soup": ["soup", "broth", "bisque", "stew"],
    "congee": ["congee", "porridge", "gruel"],
    "toast": ["toast", "open sandwich", "crumpet"],
    "bake": ["bake", "baked", "casserole", "gratin"],
}

# Standard cup measurements
STANDARD_MEASUREMENTS = {
    "1 cup": 250,      # ml
    "3/4 cup": 180,    # ml
    "2/3 cup": 160,    # ml
    "1/2 cup": 125,    # ml
    "1/3 cup": 80,     # ml
    "1/4 cup": 60,     # ml
    "1 tbsp": 15,      # ml
    "1 tsp": 5,        # ml
}

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 0 auto;
        padding: 1rem 0;
    }
    .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    .stButton>button {
        width: 200px;  /* Smaller width */
        background-color: #2ecc71;  /* Solid green color */
        color: white;  /* White text */
        border: none;  /* Remove border */
        border-radius: 5px;  /* Rounded corners */
        padding: 0.5rem 1rem;  /* Smaller padding */
        font-size: 1rem;  /* Smaller font size */
        margin: 0 auto;  /* Center the button */
        font-weight: bold;  /* Bold text */
    }
    .stButton>button:hover {
        background-color: #2ecc71;  /* Keep same green color */
        color: black;  /* Black text on hover */
    }
    .stButton>button:active {
        background-color: #2ecc71;  /* Keep same green color */
        color: black !important;  /* Force black text when pressed */
    }
    .stButton>button:focus {
        background-color: #2ecc71;  /* Keep same green color */
        color: black !important;  /* Force black text when focused */
    }
    .stButton>button:visited {
        background-color: #2ecc71;  /* Keep same green color */
        color: black !important;  /* Force black text after visited */
    }
    .stButton>button:active:focus {
        background-color: #2ecc71;  /* Keep same green color */
        color: black !important;  /* Force black text when active and focused */
    }
    .stButton>button:active:hover {
        background-color: #2ecc71;  /* Keep same green color */
        color: black !important;  /* Force black text when active and hovered */
    }
    .meal-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stSpinner > div {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    /* TEMPORARILY COMMENTED OUT: Generation time CSS
    .generation-time {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 14px;
        z-index: 1000;
    }
    */
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'meal_plan' not in st.session_state:
    st.session_state.meal_plan = None

def replace_tomato_sauce(text):
    """
    Replaces 'tomato sauce' with 'tomato-based' only when it's used generically,
    not referring to actual bottled/pasta sauces.
    """
    return re.sub(r'\btomato sauce\b', 'tomato-based', text, flags=re.IGNORECASE)

def get_user_preferences():
    try:
        # Center the title using HTML
        st.markdown("<h1 style='text-align: center;'>🍽️ BiteWise - Personalized Meal Planner 🍽️</h1>", unsafe_allow_html=True)

        # Create three columns for better layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # Use number_input with increment/decrement buttons, restricted to 1-7, default 1
            num_days = st.number_input(
                "**Number of Days**",
                min_value=1,
                max_value=7,
                value=1,
                step=1
            )
            
            # Multi-select for diet types
            diet = st.multiselect(
                "**Diet Type**",
                options=[
                    "None",
                    "Whole30",
                    "Paleo",
                    "Ketogenic (Keto)",
                    "Low-carb / High-protein",
                    "High-carb / Low-fat",
                    "Plant-based",
                    "Whole food plant-based",
                    "Vegan",
                    "Vegetarian (Lacto-ovo, Lacto, Ovo)",
                    "Pescatarian",
                    "Flexitarian",
                    "Pegan (Paleo + Vegan hybrid)",
                    "DASH (Dietary Approaches to Stop Hypertension)",
                    "MIND Diet (Mediterranean-DASH hybrid)",
                    "Intermittent Fasting (e.g. 16:8, 5:2)",
                    "Kosher",
                    "Halal",
                    "Jain",
                    "Buddhist",
                    "Seventh-day Adventist",
                    "Rastafarian Ital"
                ],
                default=[],  # Start with empty selection
                placeholder="Choose an option"
            )
            
            # If no option is selected, set to "None"
            if not diet:
                diet = ["None"]
            # If "None" is selected along with other options, clear other selections
            elif "None" in diet and len(diet) > 1:
                diet = ["None"]
            
            # Multi-select for allergies/intolerances
            allergies = st.multiselect(
                "**Allergies/Intolerances**",
                options=[
                    "No allergies or intolerances",
                    "Peanut allergy",
                    "Tree nut allergy (e.g. almond, cashew, walnut)",
                    "Shellfish allergy",
                    "Fish allergy",
                    "Egg allergy",
                    "Milk allergy (cow's milk protein: casein and/or whey)",
                    "Soy allergy",
                    "Wheat allergy",
                    "Sesame allergy",
                    "Mustard allergy",
                    "Lupin allergy",
                    "Celery allergy",
                    "Lactose intolerance",
                    "Gluten intolerance / Non-celiac gluten sensitivity",
                    "Fructose intolerance (hereditary or malabsorption)",
                    "Histamine intolerance",
                    "Salicylate sensitivity",
                    "FODMAP intolerance",
                    "Sulphite sensitivity",
                    "MSG sensitivity",
                    "Caffeine sensitivity",
                    "Alcohol intolerance",
                    "Artificial sweetener sensitivity",
                    "Food additive intolerance (e.g. preservatives, colourings)"
                ],
                default=[],  # Start with empty selection
                placeholder="Choose an option"
            )
            
            # If no option is selected, set to "No allergies or intolerances"
            if not allergies:
                allergies = ["No allergies or intolerances"]
            # If "No allergies or intolerances" is selected along with other options, clear other selections
            elif "No allergies or intolerances" in allergies and len(allergies) > 1:
                allergies = ["No allergies or intolerances"]

        with col2:
            # Multi-select for health conditions
            health_conditions = st.multiselect(
                "**Health Conditions**",
                options=[
                    "None",
                    # Metabolic & Endocrine 
                    "Type 1 Diabetes", "Type 2 Diabetes", "Prediabetes / Insulin Resistance", "Metabolic Syndrome",
                    "PCOS", "Hypothyroidism / Hashimoto's", "Hyperthyroidism / Graves'", "Adrenal Fatigue",
                    "Cushing's Syndrome", "Gestational Diabetes",
                    # Cardiovascular & Blood
                    "Hypertension", "High Cholesterol / Dyslipidemia", "Cardiovascular Disease", "Stroke prevention",
                    "Congestive Heart Failure", "Anticoagulant therapy (e.g. stable vitamin K for Warfarin)",
                    "Anaemia (Iron, B12, Folate)",
                    # Gastrointestinal
                    "Celiac Disease", "IBS (Irritable Bowel Syndrome)", "IBD (Crohn's, Ulcerative Colitis)",
                    "GERD / Acid Reflux", "Peptic Ulcers", "Diverticulosis / Diverticulitis", "Gallbladder disease",
                    "Pancreatitis", "Gastroparesis", "Liver disease / Fatty liver", "Bile acid malabsorption",
                    "SIBO (Small Intestinal Bacterial Overgrowth)",
                    # Kidney, Liver, Gout
                    "Chronic Kidney Disease (CKD)", "Kidney stones", "Nephrotic syndrome", "Gout", "Hemochromatosis",
                    "Liver Cirrhosis",
                    # Cancer & Treatment Recovery
                    "Cancer-related weight loss / Cachexia", "Neutropenic diet (immunosuppressed)",
                    "Low-residue diet (during flare-ups or treatment)",
                    # Autoimmune & Immune Conditions
                    "Lupus", "Rheumatoid Arthritis", "Multiple Sclerosis",
                    "Chronic hives / urticaria", "Mast Cell Activation Syndrome (MCAS)",
                    # Neurological & Mental Health
                    "Epilepsy (Keto for seizures)", "ADHD", "Autism Spectrum Disorder", "Depression / Anxiety",
                    "Migraine / Vestibular migraine", "Alzheimer's / Dementia", "Parkinson's Disease",
                    # Muscle, Bone, Joint
                    "Osteoporosis", "Osteopenia", "Sarcopenia / Muscle loss", "Arthritis (Osteoarthritis, Gout, RA)",
                    # Reproductive & Hormonal
                    "Endometriosis", "PMS / PMDD", "Fertility (male and female)", "Pregnancy (Trimester-specific)",
                    "Breastfeeding", "Menopause", "Erectile dysfunction",
                    # Weight & Nutrition Risk
                    "Overweight / Obesity", "Underweight / Malnutrition", "Muscle gain",
                    "Bariatric surgery (pre and post-op)", "Disordered eating / ED recovery", "Cachexia",
                    # Skin Conditions
                    "Acne", "Rosacea", "Eczema", "Psoriasis"
                ],
                default=[],  # Start with empty selection
                placeholder="Choose an option"
            )
            
            # If no option is selected, set to "None"
            if not health_conditions:
                health_conditions = ["None"]
            # If "None" is selected along with other options, clear other selections
            elif "None" in health_conditions and len(health_conditions) > 1:
                health_conditions = ["None"]

            # Multi-select for meal type
            meal_type = st.multiselect(
                "**Meal Type Preference**",
                options=["All", "Breakfast", "Lunch", "Dinner"],
                default=[],  # Start with empty selection
                placeholder="Choose an option"
            )
            
            # If no option is selected, set to "All"
            if not meal_type:
                meal_type = ["All"]
            # If "All" is selected along with other options, clear other selections
            elif "All" in meal_type and len(meal_type) > 1:
                meal_type = ["All"]

            # Multi-select for cuisine
            cuisine = st.multiselect(
                "**Preferred Cuisine**",
                options=[
                    "All",
                    "Mediterranean",
                    "Thai",
                    "Chinese",
                    "Japanese",
                    "Korean",
                    "Vietnamese",
                    "Indian",
                    "Middle Eastern",
                    "Latin American / Mexican",
                    "African",
                    "Nordic / Scandinavian",
                    "Traditional Australian / British / American",
                    "Eastern European",
                    "Caribbean"
                ],
                default=["Traditional Australian / British / American"],  # Set default to Traditional Australian / British / American
                placeholder="Choose an option"
            )
            
            # If no option is selected, set to "All"
            if not cuisine:
                cuisine = ["All"]
            # If "All" is selected along with other options, clear other selections
            elif "All" in cuisine and len(cuisine) > 1:
                cuisine = ["All"]

        with col3:
            budget = st.selectbox(
                "**Budget**",
                options=["Tight budget ($3-$7)", "Moderate budget ($8-$15)", 
                        "Generous budget ($16-$30)", "No budget constraints ($31+)"],
                index=1  # Default to "Moderate budget"
            )
                        
            time_constraint = st.selectbox( 
                "**Available Time for Cooking**",
                options=[
                    "Busy schedule (less than 15 minutes)",  
                    "Moderate schedule (15 to 30 minutes)",  
                    "Busy on some days (30 to 45 minutes)",  
                    "Flexible schedule (45 to 60 minutes)",  
                    "No constraints (more than 60 minutes)"
                ],
                index=1
            )

            # Map UI values to standardized cooking time ranges
            time_mapping = {
                "Busy schedule (less than 15 minutes)": "no more than 15 minutes",
                "Moderate schedule (15 to 30 minutes)": "between 15 and 30 minutes",
                "Busy on some days (30 to 45 minutes)": "between 30 and 45 minutes",
                "Flexible schedule (45 to 60 minutes)": "between 45 and 60 minutes",
                "No constraints (more than 60 minutes)": "no time limit"
            }

            # Convert to standardized format immediately
            time_constraint = time_mapping.get(time_constraint, "between 15 and 30 minutes")
                        
            # Changed from selectbox to number_input for serving size
            serving_size = st.number_input(
                "**Number of Servings per Meal**",
                min_value=1,
                max_value=8,
                value=2,  # Default to 2 servings
                step=1
            )

        # Create a dictionary with all preferences
        preferences = {
            "num_days": int(num_days),  # Ensure it's an integer
            "diet": str(diet),  # Ensure it's a string
            "health_conditions": str(health_conditions),  # Ensure it's a string
            "meal_type": str(meal_type),  # Ensure it's a string
            "cuisine": str(cuisine),  # Ensure it's a string
            "budget": str(budget),  # Ensure it's a string
            "time_constraint": time_constraint,  # Already in standardized format
            "serving_size": int(serving_size),  # Ensure it's an integer
            "allergies": str(allergies)  # Ensure it's a string
        }

        # Store preferences in session state
        st.session_state.user_preferences = preferences

        return preferences
    except Exception as e:
        st.error(f"Error getting user preferences: {str(e)}")
        return None

def build_diet_rules_block(diet_type):
    diet_rules = ""
    if diet_type != "None":
        if isinstance(diet_type, list):
            for diet in diet_type:
                if diet in DIET_RESTRICTIONS:
                    restricted_items = DIET_RESTRICTIONS[diet]
                    diet_rules += f"\nIMPORTANT - {diet} Diet Rules:\n"
                    diet_rules += f"- DO NOT include any of the following: {', '.join(restricted_items)}.\n"
                    diet_rules += f"- These ingredients are strictly forbidden and GPT must never include or mention them.\n"
        else:
            if diet_type in DIET_RESTRICTIONS:
                restricted_items = DIET_RESTRICTIONS[diet_type]
                diet_rules += f"\nIMPORTANT - {diet_type} Diet Rules:\n"
                diet_rules += f"- DO NOT include any of the following: {', '.join(restricted_items)}.\n"
                diet_rules += f"- These ingredients are strictly forbidden and GPT must never include or mention them.\n"
    return diet_rules

def build_health_restrictions_block(health_conditions):
    rules = ""
    for condition in health_conditions:
        condition_clean = condition.strip()
        if condition_clean in HEALTH_RESTRICTIONS:
            restricted = HEALTH_RESTRICTIONS[condition_clean]
            rules += f"\nHEALTH: {condition_clean}\n"
            rules += f"- Avoid: {', '.join(restricted)}\n"
    return rules.strip()

def build_allergy_block(allergies):
    rules = ""
    for allergy in allergies:
        allergy_clean = allergy.strip()
        if allergy_clean.lower() not in ["no allergies", "no allergies or intolerances"]:
            keywords = ALLERGEN_KEYWORDS.get(allergy_clean, [])
            if keywords:
                rules += f"\nALLERGY: {allergy_clean}\n"
                rules += f"- STRICTLY AVOID: {', '.join(keywords)}\n"
    return rules.strip()

async def is_similar_recipe_with_embedding(embedding):
    if recipe_index.ntotal == 0:
        return False
    D, _ = recipe_index.search(np.array([embedding], dtype="float32"), k=1)
    return D[0][0] < 0.15

async def save_recipe_embedding_with_embedding(name, ingredients, embedding):
    if embedding is None or len(embedding) != 1536:
        print("❌ Invalid embedding, skipping save.")
        return

    recipe_index.add(np.array([embedding], dtype="float32"))
    recipe_names.append(name)

    # Don't write to disk here
    print(f"✅ Saved recipe embedding in-memory for '{name}'")

st.markdown("""
    <style>
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

async def generate_meal(meal_type, day, prompt, cuisine="All", recipe_name=""):
    try:
        await init_http_session()

        data = {
            "model": "gpt-4o-mini",
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1
        }

        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        }

        # ✅ Capture the stream into recipe_text
        recipe_text = ""

        def token_stream():
            q = queue.Queue()

            async def fetch():
                async with aiohttp.ClientSession() as sess:
                    async with sess.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers) as response:
                        async for line in response.content:
                            if line:
                                decoded_line = line.decode("utf-8").strip()
                                if decoded_line.startswith("data: "):
                                    decoded_line = decoded_line.replace("data: ", "")
                                    if decoded_line == "[DONE]":
                                        break
                                    try:
                                        parsed = json.loads(decoded_line)
                                        token = parsed["choices"][0]["delta"].get("content", "")
                                        q.put(token)
                                    except Exception as e:
                                        print(f"[STREAM ERROR] Could not parse: {decoded_line} — {e}")
                q.put(None)

            threading.Thread(target=lambda: asyncio.run(fetch()), daemon=True).start()

            def generator():
                while True:
                    token = q.get()
                    if token is None:
                        break
                    nonlocal recipe_text
                    recipe_text += token
                    yield token

            return generator()

        recipe_text = ""
        return (meal_type, day, token_stream)

    except Exception as e:
        print(f"[ERROR] {meal_type} on Day {day} failed: {str(e)}")
        return None

import threading
import queue

import threading
import queue

def safe_stream_and_capture(token_gen):
    q = queue.Queue()
    recipe_text_holder = {"text": ""}

    def run():
        for token in token_gen:
            q.put(token)
            recipe_text_holder["text"] += token
        q.put(None)

    def live_stream():
        while True:
            token = q.get()
            if token is None:
                break
            yield token

    threading.Thread(target=run).start()
    return live_stream(), lambda: recipe_text_holder["text"]

async def save_all_embeddings():
    recipes = st.session_state.recipes_to_embed
    if not recipes:
        return

    texts = [f"{r['name']}\n{r['ingredients']}" for r in recipes]

    start_embed = time.time()
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    print(f"🧠 Embedding generation time: {time.time() - start_embed:.2f} seconds")

    for i, recipe in enumerate(recipes):
        embedding = response.data[i].embedding
        if len(embedding) == 1536:
            recipe_index.add(np.array([embedding], dtype="float32"))
            recipe_names.append(recipe["name"])
        else:
            print(f"⚠️ Invalid embedding for {recipe['name']}")

    print(f"✅ Batch-saved {len(recipes)} recipe embeddings.")

def clean_recipe_text(text):
    """Clean recipe text for comparison."""
    # Remove measurements and quantities
    text = re.sub(r'\d+\s*(?:tbsp|tsp|cup|cups|g|kg|ml|l|oz|lb|inch|cm|mm|°C|°F|min|hour|hours|minute|minutes|second|seconds)', '', text)
    # Remove common words and ingredients
    common_words = ['ingredients', 'instructions', 'method', 'steps', 'serves', 'total time', 'prep time', 'cook time', 'time', 'serving', 'servings']
    for word in common_words:
        text = text.replace(word, '')
    # Remove extra spaces and convert to lowercase
    text = ' '.join(text.split()).lower()
    return text

from prompts import DIETARY_REQUIREMENTS

import ast
import re

def is_recipe_safe(recipe_text, ingredients_text, user_prefs):
    text_to_check = recipe_text.lower()


    # ✅ 1. Allergy Check — FULL FIXED BLOCK
    allergies_raw = user_prefs.get("allergies", "")
    try:
        allergies = ast.literal_eval(allergies_raw) if isinstance(allergies_raw, str) else allergies_raw
        if not isinstance(allergies, list):
            allergies = []
    except Exception as e:
        print(f"[ERROR] Failed to parse allergies: {e}")
        allergies = []

    if not any(a.strip().lower() == "no allergies or intolerances" for a in allergies):
        for allergy in allergies:
            allergy_clean = allergy.strip().title()
            keywords = ALLERGEN_KEYWORDS.get(allergy_clean, [])
            for keyword in keywords:
                pattern = rf"\b{re.escape(keyword.lower())}s?\b"
                if re.search(pattern, text_to_check):
                    print(f"[ALLERGY BLOCKED] Matched '{keyword}' for allergy '{allergy_clean}'")
                    return False

    # ✅ 2. Diet Restriction Check
    diet_raw = user_prefs.get("diet", "none")
    try:
        diet = ast.literal_eval(diet_raw) if isinstance(diet_raw, str) and diet_raw.startswith("[") else [diet_raw]
    except Exception:
        diet = [diet_raw]

    for d in diet:
        if d in DIET_RESTRICTIONS:
            for keyword in DIET_RESTRICTIONS[d]:
                if re.search(rf"\b{re.escape(keyword)}\b", text_to_check):
                    print(f"[DIET VIOLATION] Matched '{keyword}' for diet '{d}'")
                    return False

    # ✅ 3. Health Condition Check
    health_raw = user_prefs.get("health_conditions", "")
    try:
        health_conditions = ast.literal_eval(health_raw) if isinstance(health_raw, str) and health_raw.startswith("[") else [health_raw]
    except Exception:
        health_conditions = [health_raw]

    for condition in health_conditions:
        condition = condition.strip().lower()
        if condition in ["none", ""]:
            continue
        keywords = HEALTH_RESTRICTIONS.get(condition, [])
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", text_to_check):
                print(f"[HEALTH VIOLATION] Matched '{keyword}' for condition '{condition}'")
                return False

    return True

import unicodedata

def is_title_safe(title: str, allergy_keywords: dict, selected_allergies: list) -> bool:
    title_lower = title.lower()
    for allergy in selected_allergies:
        if allergy in allergy_keywords:
            for keyword in allergy_keywords[allergy]:
                if re.search(rf"\b{re.escape(keyword)}\b", title_lower):
                    return False  # Block this title
    return True


def get_structure_type(title):
    title_lower = title.lower()
    for structure, keywords in STRUCTURE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in title_lower:
                return structure
    return None

# ========== FAISS Embedding Utils ==========
async def get_embedding(text):
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

async def get_recipe_embedding(name, ingredients):
    combined_text = f"{name}\n{ingredients}"
    return await get_embedding(combined_text)

async def save_recipe_embedding(name, ingredients):
    embedding = await get_recipe_embedding(name, ingredients)

    if embedding is None or len(embedding) != 1536:
        print("❌ Invalid embedding, skipping save.")
        return

    recipe_index.add(np.array([embedding], dtype="float32"))
    recipe_names.append(name)

    # Save index and names to disk
    faiss.write_index(recipe_index, INDEX_FILE)
    with open(NAMES_FILE, "wb") as f:
        pickle.dump(recipe_names, f)

    print(f"✅ Saved recipe embedding for '{name}'")

def is_title_allowed_for_diet(recipe_name, diet_list):
    restricted_keywords = set()
    for diet in diet_list:
        if diet in DIET_RESTRICTIONS:
            restricted_keywords.update(DIET_RESTRICTIONS[diet])

    recipe_name_lower = recipe_name.lower()
    for keyword in restricted_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", recipe_name_lower):
            print(f"[TITLE REJECTION] Matched '{keyword}' in recipe title for diet: {diet_list}")
            return False
    return True

import re

def parse_recipe(recipe_text):
    try:
        # Strip markdown formatting and clean whitespace
        cleaned_text = recipe_text.replace("**", "").replace("__", "").strip()

        # Match title pattern flexibly (e.g., "Day 1 - Breakfast - Name | Breakfast")
        title_match = re.search(
            r"(?:Day\s*\d+\s*-\s*)?(?:Breakfast|Lunch|Dinner)?\s*-\s*(.*?)\s*\|", 
            cleaned_text, 
            re.IGNORECASE
        )
        name = title_match.group(1).strip() if title_match else None

        # Match Ingredients section robustly (accepts "Ingredients", "Ingredient list", case-insensitive)
        ingredients_match = re.search(
            r"(?i)(?:ingredients|ingredient list)\s*[:\-]?\s*(.*?)(?:\n\s*(instructions|description|total time|serves)\b|$)", 
            cleaned_text, 
            re.DOTALL
        )
        ingredients_raw = ingredients_match.group(1).strip() if ingredients_match else None

        if not name or not ingredients_raw:
            print("❌ parse_recipe() failed: Missing recipe name or ingredients.")
            print("----- RAW TEXT BEGIN -----")
            print(recipe_text)
            print("----- RAW TEXT END -------")
            return None

        # Extract each line, remove bullets, hyphens, etc.
        lines = [line.strip("-• ").strip() for line in ingredients_raw.splitlines() if line.strip()]
        ingredients = ", ".join(lines)

        return {
            "name": name,
            "ingredients": ingredients
        }

    except Exception as e:
        print(f"[parse_recipe] Exception: {e}")
        return None

async def is_similar_recipe(name, ingredients):
    if recipe_index.ntotal == 0:
        return False

    combined_text = f"{name}\n{ingredients}"
    embedding = await get_embedding(combined_text)
    D, _ = recipe_index.search(np.array([embedding], dtype="float32"), k=1)
    return D[0][0] < 0.15

def calculate_similarity(recipe1, recipe2):
    """Calculate similarity between two recipes using multiple criteria."""
    # Text similarity for recipe names
    name_similarity = SequenceMatcher(None, recipe1['name'].lower(), recipe2['name'].lower()).ratio()
    
    # Extract ingredients and convert to sets
    ingredients1 = set(recipe1['ingredients'].lower().split('\n'))
    ingredients2 = set(recipe2['ingredients'].lower().split('\n'))
    
    # Calculate ingredient similarity
    if ingredients1 and ingredients2:
        ingredient_similarity = len(ingredients1.intersection(ingredients2)) / max(len(ingredients1), len(ingredients2))
    else:
        ingredient_similarity = 0.0
    
    # Extract cooking methods
    methods1 = set([word.lower() for word in recipe1['instructions'].split() if word in ['bake', 'fry', 'grill', 'roast', 'steam', 'boil', 'stir-fry']])
    methods2 = set([word.lower() for word in recipe2['instructions'].split() if word in ['bake', 'fry', 'grill', 'roast', 'steam', 'boil', 'stir-fry']])
    
    # Calculate cooking method similarity
    if methods1 and methods2:
        method_similarity = len(methods1.intersection(methods2)) / max(len(methods1), len(methods2))
    else:
        method_similarity = 0.0
    
    # Calculate weighted average of similarities
    weights = {
        'name': 0.4,
        'ingredients': 0.3,
        'methods': 0.3
    }
    
    total_similarity = (
        name_similarity * weights['name'] +
        ingredient_similarity * weights['ingredients'] +
        method_similarity * weights['methods']
    )
    
    return total_similarity

def is_recipe_unique(new_recipe, existing_recipes, threshold=0.3):
    """Check if a recipe is unique compared to existing recipes."""
    for existing_recipe in existing_recipes:
        similarity = calculate_similarity(new_recipe, existing_recipe)
        if similarity > threshold:
            return False
    return True

def choose_unused_title(meal_type, cuisine, attempted_titles):
    """
    Select a unique recipe title that hasn't already been tried.
    Pulls from your AUTHENTIC_RECIPE_NAMES dictionary.
    """
    meal_type = meal_type.lower()

    available_titles = [
        title for title in AUTHENTIC_RECIPE_NAMES.get(cuisine, [])
        if title.lower().endswith(f"| {meal_type}") and title not in attempted_titles
    ]

    if not available_titles:
        return None  # No more titles left to try

    return random.choice(available_titles)

if "generated_recipes" in st.session_state:
    st.session_state.generated_recipes.clear()
else:
    st.session_state.generated_recipes = []

import queue

def stream_recipe(recipe_text):
    q = queue.Queue()
    for char in recipe_text:
        q.put(char)
    q.put(None)

    def generator():
        while True:
            token = q.get()
            if token is None:
                break
            yield token
    return generator()

import queue
import threading
import streamlit as st

import queue
import threading

def stream_and_buffer(token_gen):
    q = queue.Queue()
    recipe_text_holder = {"text": ""}

    def producer():
        for token in token_gen:
            q.put(token)
            recipe_text_holder["text"] += token
        q.put(None)

    def streamer():
        while True:
            token = q.get()
            if token is None:
                break
            yield token

    threading.Thread(target=producer).start()
    return streamer(), lambda: recipe_text_holder["text"]

def stream_with_validation(token_gen, validate_callback):
    q = queue.Queue()
    full_text_holder = {"text": ""}
    is_valid = {"passed": False}

    # Background producer thread
    def producer():
        for token in token_gen:
            full_text_holder["text"] += token
            q.put(token)
        q.put(None)

    # Live generator stream to UI
    def live_stream():
        while True:
            token = q.get()
            if token is None:
                break
            yield token

    # Kick off producer thread
    threading.Thread(target=producer).start()

    # Return generator and validator callback access
    return live_stream(), lambda: full_text_holder["text"]

async def generate_meal_plan(user_prefs):
    stream_container = st.container()  # placeholder for streaming
    try:
        # --- Validate user prefs ---
        if not isinstance(user_prefs, dict):
            raise ValueError("user_prefs must be a dictionary")

        required_keys = ['num_days', 'diet', 'allergies', 'health_conditions',
                         'meal_type', 'cuisine', 'budget', 'time_constraint', 'serving_size']
        for key in required_keys:
            if key not in user_prefs:
                raise KeyError(f"Missing required preference: {key}")

        # --- Init session state trackers ---
        st.session_state.generated_recipes = []  # Store all generated recipes
        st.session_state.used_recipe_names = set()  # Track used recipe names (to avoid duplicates)
        st.session_state.meal_types_used = set()
        st.session_state.cuisines_used = set()
        st.session_state.cuisine_distribution = {}
        st.session_state.structure_counts = {}

        if "structure_counts" not in st.session_state:
            st.session_state.structure_counts = {}

        if 'recipes_to_embed' not in st.session_state:
            st.session_state.recipes_to_embed = []
        else:
            st.session_state.recipes_to_embed.clear()

        # --- Extract and prepare preferences ---
        num_days = user_prefs['num_days']
        diet_type = user_prefs['diet']
        allergies = user_prefs['allergies']
        health_conditions = user_prefs['health_conditions']
        raw_meal_types = user_prefs.get("meal_type", "")
        meal_types = parse_list(raw_meal_types)
        if not meal_types or raw_meal_types in ["", "Choose an option", None] or "All" in meal_types:
            meal_types = ["Breakfast", "Lunch", "Dinner"]

        raw_cuisine = user_prefs.get("cuisine", "")
        cuisine_list = parse_list(raw_cuisine)
        if not cuisine_list or raw_cuisine in ["", "Choose an option", None] or "All" in cuisine_list:
            cuisine_list = [
                "Mediterranean", "Latin American / Mexican", "Indian", "Japanese",
                "Chinese", "Middle Eastern", "Vietnamese",
                "Korean", "Traditional Australian / British / American", "African"
            ]
        random.shuffle(cuisine_list)

        budget = user_prefs['budget']
        cooking_time = user_prefs['time_constraint']
        servings = user_prefs['serving_size']
        diet_list = [d.strip().lower() for d in parse_list(diet_type)]

        if not isinstance(user_prefs, dict):
            raise ValueError("user_prefs must be a dictionary")

        required_keys = ['num_days', 'diet', 'allergies', 'health_conditions',
                         'meal_type', 'cuisine', 'budget', 'time_constraint', 'serving_size']

        for key in required_keys:
            if key not in user_prefs:
                raise KeyError(f"Missing required preference: {key}")

        # Cache frequently used values
        num_days = user_prefs['num_days']
        diet_type = user_prefs['diet']
        allergies = user_prefs['allergies']
        health_conditions = user_prefs['health_conditions']
        meal_type = user_prefs['meal_type']
        cuisine = user_prefs['cuisine']
        budget = user_prefs['budget']
        cooking_time = user_prefs['time_constraint']  # Already in standardized format
        servings = user_prefs['serving_size']

        # Clean lists from strings
        diet_list = [d.strip().lower() for d in parse_list(diet_type)]
        allergy_list = [a.strip() for a in parse_list(allergies)]
        health_list = [h.strip().lower() for h in parse_list(health_conditions)]

        formatted_prefs = {
            'num_days': num_days,
            'diet': diet_type,
            'allergies': allergies,
            'health_conditions': health_conditions,
            'meal_type': meal_type,
            'cuisine': cuisine,
            'budget': budget,
            'time_constraint': cooking_time,  # Use the standardized format directly
            'serving_size': servings,
            'diet_list': diet_list,
            'allergy_list': allergy_list,
            'health_list': health_list
        }

        health_requirements = IMPORTANT_RULES if health_conditions != "None" else ""

        diet_rules = ""
        if diet_type != "None":
            if isinstance(diet_type, list):
                for diet in diet_type:
                    if diet in DIET_RESTRICTIONS:
                        restricted_items = DIET_RESTRICTIONS[diet]
                        diet_rules += f"\nIMPORTANT - {diet} Diet Rules:\n"
                        diet_rules += f"- DO NOT include: {', '.join(restricted_items)}.\n"
            else:
                if diet_type in DIET_RESTRICTIONS:
                    restricted_items = DIET_RESTRICTIONS[diet_type]
                    diet_rules += f"\nIMPORTANT - {diet_type} Diet Rules:\n"
                    diet_rules += f"- DO NOT include: {', '.join(restricted_items)}.\n"

        dietary_requirements = f"""

        IMPORTANT: Strict dietary filtering enabled.

        {diet_rules}

        DO NOT include restricted ingredients even in small amounts.
        GPT must never hallucinate or bypass restrictions.
        Use culturally appropriate alternatives only if necessary.

        Additional Requirements:
        - Allergies/Intolerances: {allergies}
        - Health Conditions: {health_conditions}
        - Budget: {budget}
        - Time Constraint: {cooking_time}  # Use the standardized format directly
        - Serving Size: {servings} people

        IMPORTANT: The recipe MUST strictly follow all dietary requirements listed above.
        If there are conflicts between different diet types, prioritize the most restrictive requirements.
        """

        budget_requirements = {
            "Tight budget": """
            Use these budget-friendly ingredients:
            - Proteins: eggs, lentils, chickpeas, beans, canned tuna
            - Vegetables: potatoes, onions, carrots, cabbage, frozen vegetables
            - Grains: rice, pasta, oats
            - Avoid: expensive cuts of meat and exotic ingredients
            """,
            "Moderate budget": """
            Use these moderately priced ingredients:
            - Proteins: chicken thighs, chicken breasts, ground beef, tofu, eggs
            - Vegetables: seasonal vegetables, frozen vegetables
            - Grains: rice, pasta, bread
            - Include some fresh herbs and spices
            """,
            "Generous budget": """
            Use these premium ingredients:
            - Proteins: fish, lean meats, specialty proteins
            - Vegetables: fresh seasonal vegetables, specialty produce
            - Grains: quinoa, specialty grains
            - Include premium ingredients and fresh herbs
            """
        }.get(budget, "")

        measurement_requirements = """
        Use these standard measurements:
        - 1 cup = 250ml
        - 1 tablespoon = 15ml
        - 1 teaspoon = 5ml
        - Use metric measurements for liquids
        - Use standard household measurements for solids
        """

        available_ingredients = []
        for category, items in CATEGORIES.items():
            if any(budget.lower() in category.lower() for budget in ["Budget", "Affordable"]):
                available_ingredients.extend(items)

        cuisine_prefs = eval(user_prefs['cuisine'])
        if not cuisine_prefs or "All" in cuisine_prefs:
            available_cuisines = [
                "Mediterranean", "Latin American / Mexican", "Indian", "Japanese",
                "Chinese", "Middle Eastern", "Vietnamese",
                "Korean", "Traditional Australian / British / American", "African"
            ]
            available_cuisines = [c for c in available_cuisines if c != "Thai"]
            random.shuffle(available_cuisines)
        else:
            available_cuisines = cuisine_prefs.copy()
            random.shuffle(available_cuisines)

        semaphore = asyncio.Semaphore(10)
            
        async def limited_generate(meal_type, day, prompt, cuisine, recipe_name):
            async with semaphore:
                return await generate_meal(meal_type, day, prompt, cuisine, recipe_name)

        tasks = []

        cuisine_index = 0
        
        for day in range(1, num_days + 1):
            st.session_state.cuisine_distribution[day] = {
                'breakfast': None,
                'lunch': None,
                'dinner': None
            }

            if "All" in meal_type:
                meal_types = ["Breakfast", "Lunch", "Dinner"]
            else:
                meal_types = [m for m in ["Breakfast", "Lunch", "Dinner"] if m in meal_type]

            day_meal_prompts = []  # [(meal_type, prompt, cuisine, recipe_name)]
            meal_title_map = {}    # {meal_type: [titles]}

            # Start processing for each meal in the meal_types
            for meal in meal_types:
                selected_cuisine = available_cuisines[cuisine_index % len(available_cuisines)]
                cuisine_index += 1
                st.session_state.cuisine_distribution[day][meal.lower()] = selected_cuisine

                # Get authentic recipes for the selected cuisine
                authentic_recipes = AUTHENTIC_RECIPE_NAMES.get(selected_cuisine, [])
                meal_recipes = [
                    r for r in authentic_recipes
                    if meal.lower() in r.lower() and is_title_allowed_for_diet(r, diet_list)
                ]

                if not meal_recipes:
                    fallback = f"{selected_cuisine} {meal} - GPT Generated Fallback"
                    meal_recipes = [fallback]

                random.shuffle(meal_recipes)

                retry_titles = []
                for recipe_name in meal_recipes:
                    if recipe_name in st.session_state.used_recipe_names:
                        continue
                    retry_titles.append(recipe_name)
                meal_title_map[meal] = retry_titles

                found = False
                for recipe_name in retry_titles:
                    if recipe_name in st.session_state.used_recipe_names:
                        continue

                    if not is_title_safe(recipe_name, ALLERGEN_KEYWORDS, allergy_list):
                        print(f"[BLOCKED] Title '{recipe_name}' contains allergen — skipping.")
                        continue

                    prompt = get_meal_prompt(
                        meal_type=meal,
                        day=day,
                        user_prefs=formatted_prefs,
                        health_requirements=health_requirements,
                        cuisine_requirements=f"""
                        {selected_cuisine} Cuisine Requirements:
                        - Use authentic {selected_cuisine} ingredients and methods
                        - Follow {selected_cuisine} cultural traditions
                        - Use traditional {selected_cuisine} dishes
                        - Avoid mixing with other cuisines
                        - Ensure dish is recognizably {selected_cuisine}

                        IMPORTANT: Use the exact recipe name: \"{recipe_name}\". No rephrasing.
                        """ + dietary_requirements + budget_requirements + measurement_requirements,
                        available_ingredients=available_ingredients,
                        authentic_recipes=[recipe_name]
                    )

                    result = await limited_generate(meal, day, prompt, selected_cuisine, recipe_name)

                    if isinstance(result, tuple):
                        _, _, token_gen_func = result
                        token_gen = token_gen_func()

                        # Start streaming + capturing at the same time
                        live_stream, get_full_text = stream_with_validation(token_gen, None)

                        # Stream to UI container
                        stream_container.write_stream(live_stream)

                        # Wait for full text to complete (safely)
                        start = time.time()
                        timeout = 40  # seconds
                        recipe_text = ""

                        while time.time() - start < timeout:
                            recipe_text = get_full_text()
                            if "Serves" in recipe_text and "Instructions" in recipe_text and len(recipe_text) > 400:
                                break
                            time.sleep(0.2)

                        # If still empty → skip
                        if not recipe_text.strip():
                            print("[SKIPPED] Recipe generation timed out or failed.")
                            continue

                        # Validate full recipe
                        parsed = parse_recipe(recipe_text)
                        if not parsed:
                            print(f"[REJECTED] could not parse")
                            continue

                        if not is_recipe_safe(recipe_text, parsed["ingredients"], formatted_prefs):
                            print(f"[REJECTED] due to allergy/diet violation")
                            continue

                        meal_key = (day, meal)
                        if meal_key in [(r["day"], r["meal_type"]) for r in st.session_state.generated_recipes]:
                            print(f"[REJECTED] duplicate {meal_key}")
                            continue

                        # Store it ✅
                        st.session_state.generated_recipes.append({
                            "day": day,
                            "meal_type": meal,
                            "recipe": recipe_text
                        })
                        st.session_state.used_recipe_names.add(recipe_name)
                        found = True
                        break

    finally:
        await save_all_embeddings()
        faiss.write_index(recipe_index, INDEX_FILE)
        with open(NAMES_FILE, "wb") as f:
            pickle.dump(recipe_names, f)
        print("✅ FAISS index and names saved to disk at end.")

def display_meal_plan(meal_plan):
    #st.title("Your Personalized Meal Plan")
    
    # Format meal plan for display and download
    formatted_meal_plan = ""
    for meal in meal_plan:
        # Ensure that both 'day' and 'meal_type' are properly provided in `meal_plan`
        day = meal.get('day')  # Assuming 'day' is part of the meal plan
        
        # Skip meal if it's already added for the specific day and meal type
        if any(r['day'] == day and r['meal_type'] == meal['meal_type'] for r in st.session_state.generated_recipes):
            print(f"[SKIP] {meal['meal_type']} already added for Day {day}")
            continue
        
        # Format the recipe content
        recipe_content = meal['recipe']
        
        # Split the recipe into sections
        sections = recipe_content.split("\n")
        formatted_sections = []
        
        for section in sections:
            if section.strip().lower().startswith("ingredients:"):
                # Format ingredients section
                ingredients_text = section.replace("Ingredients:", "").strip()
                # Split by bullet points and clean up
                ingredients = [ing.strip() for ing in ingredients_text.split("•") if ing.strip()]
                formatted_ingredients = "Ingredients:\n"
                for ingredient in ingredients:
                    formatted_ingredients += f"• {ingredient}\n"
                formatted_sections.append(formatted_ingredients)
            elif section.strip().lower().startswith("instructions:"):
                # Add extra newline before instructions
                formatted_sections.append("\nInstructions:")
            else:
                formatted_sections.append(section)
        
        # Join the sections back together with proper spacing
        formatted_recipe = "\n".join(formatted_sections)
        
        # Add to the meal plan
        formatted_meal_plan += "\n"
        formatted_meal_plan += formatted_recipe
        formatted_meal_plan += "\n"
    
    # Display the formatted meal plan
    st.markdown(formatted_meal_plan)
    
def extract_ingredients(recipe_text):
    """Extract main ingredients from recipe text."""
    ingredients = set()
    
    # Look for ingredients section
    if "Ingredients:" in recipe_text:
        ingredients_section = recipe_text.split("Ingredients:")[1].split("Instructions:")[0]
        
        # Split into lines and process each line
        for line in ingredients_section.split('\n'):
            line = line.strip()
            if line and not line.startswith('•') and not line.startswith('-'):
                # Remove quantities and units
                ingredient = line.split('(')[0].strip()
                ingredient = ingredient.split(',')[0].strip()
                ingredient = ingredient.split('or')[0].strip()
                if ingredient:
                    ingredients.add(ingredient.lower())
    
    return ingredients

def extract_cooking_methods(recipe_text):
    """Extract cooking methods from recipe text."""
    cooking_methods = set()
    
    # Common cooking methods to look for
    methods = {
        'bake', 'boil', 'broil', 'grill', 'fry', 'sauté', 'sauté', 'roast', 
        'steam', 'stir-fry', 'simmer', 'poach', 'braise', 'blanch', 'deep-fry',
        'pan-fry', 'sear', 'toast', 'warm', 'heat', 'cook', 'prepare'
    }
    
    # Look for cooking methods in instructions
    if "Instructions:" in recipe_text:
        instructions = recipe_text.split("Instructions:")[1]
        
        # Check each line for cooking methods
        for line in instructions.split('\n'):
            line = line.lower()
            for method in methods:
                if method in line:
                    cooking_methods.add(method)
    
    return cooking_methods

def extract_sauces(recipe_text):
    """Extract sauces and seasonings from recipe text."""
    sauces = set()
    
    # Common sauces and seasonings to look for
    sauce_types = {
        'soy sauce', 'fish sauce', 'oyster sauce', 'hoisin sauce', 'teriyaki sauce',
        'worcestershire sauce', 'barbecue sauce', 'hot sauce', 'chili sauce',
        'tahini', 'harissa', 'sambal oelek', 'gochujang', 'miso', 'pesto',
        'vinaigrette', 'dressing', 'marinade', 'sauce', 'glaze', 'dip'
    }
    
    # Look for sauces in ingredients and instructions
    if "Ingredients:" in recipe_text:
        ingredients_section = recipe_text.split("Ingredients:")[1].split("Instructions:")[0]
        
        # Check ingredients section
        for line in ingredients_section.split('\n'):
            line = line.lower()
            for sauce in sauce_types:
                if sauce in line:
                    sauces.add(sauce)
    
    if "Instructions:" in recipe_text:
        instructions = recipe_text.split("Instructions:")[1]
        
        # Check instructions section
        for line in instructions.split('\n'):
            line = line.lower()
            for sauce in sauce_types:
                if sauce in line:
                    sauces.add(sauce)
    
    return sauces

def extract_proteins(recipe_text):
    # Extract protein sources from recipe text
    proteins = set()
    protein_keywords = [
        'chicken', 'beef', 'pork', 'lamb', 'fish', 'salmon', 'tuna', 'shrimp',
        'eggs', 'tofu', 'tempeh', 'chickpeas', 'lentils', 'beans', 'quinoa',
        'yogurt', 'cheese', 'milk', 'protein powder'
    ]
    
    # Convert recipe text to lowercase for case-insensitive matching
    recipe_lower = recipe_text.lower()
    
    # Look for protein keywords in the recipe text
    for keyword in protein_keywords:
        if keyword in recipe_lower:
            proteins.add(keyword)
    
    return proteins

def extract_vegetables(recipe_text):
    # Extract vegetables from recipe text
    vegetables = set()
    vegetable_keywords = [
        'spinach', 'kale', 'lettuce', 'broccoli', 'cauliflower', 'carrots',
        'zucchini', 'squash', 'peppers', 'onions', 'garlic', 'tomatoes',
        'cucumber', 'celery', 'mushrooms', 'asparagus', 'green beans',
        'sweet potato', 'potato', 'pumpkin'
    ]
    
    # Convert recipe text to lowercase for case-insensitive matching
    recipe_lower = recipe_text.lower()
    
    # Look for vegetable keywords in the recipe text
    for keyword in vegetable_keywords:
        if keyword in recipe_lower:
            vegetables.add(keyword)
    
    return vegetables

def extract_grains(recipe_text):
    # Extract grains from recipe text
    grains = set()
    grain_keywords = [
        'rice', 'quinoa', 'oats', 'barley', 'couscous', 'pasta', 'bread',
        'wheat', 'buckwheat', 'millet', 'amaranth', 'teff', 'sorghum'
    ]
    
    # Convert recipe text to lowercase for case-insensitive matching
    recipe_lower = recipe_text.lower()
    
    # Look for grain keywords in the recipe text
    for keyword in grain_keywords:
        if keyword in recipe_lower:
            grains.add(keyword)
    
    return grains

import streamlit as st

async def main():
    try:
        user_prefs = get_user_preferences()

        if user_prefs is None:
            st.error("Failed to get user preferences. Please try again.")
            return

        # Initialize success message flag if not set
        if 'show_success_message' not in st.session_state:
            st.session_state.show_success_message = False

        # Placeholder for success message so it can be cleared
        success_placeholder = st.empty()

        col1, col2, col3 = st.columns([0.05, 2.9, 0.05])
        with col2:
            if st.button("Generate Meal Plan"):
                with st.spinner("Generating Your Personalized Meal Plan..."):
                    try:
                        # ✅ FULL STATE RESET
                        st.session_state.used_recipe_names = set()
                        st.session_state.generated_recipes = []
                        st.session_state.meal_types_used = set()
                        st.session_state.cuisines_used = set()
                        st.session_state.cuisine_distribution = {}
                        st.session_state.structure_counts = {}
                        st.session_state.recipes_to_embed = []
                        st.session_state.show_success_message = False
                        st.session_state.meal_plan = None  # ✅ CLEAR PREVIOUS PLAN

                        # ✅ RESET FAISS MEMORY + FILES
                        recipe_index.reset()
                        recipe_names.clear()
                        if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
                        if os.path.exists(NAMES_FILE): os.remove(NAMES_FILE)

                        # ✅ GENERATE
                        start_time = time.time()
                        meal_plan = await generate_meal_plan(user_prefs)
                        end_time = time.time()
                        print(f"⏱️ Total generation time: {end_time - start_time:.2f} seconds")

                        st.session_state.meal_plan = meal_plan

                        if meal_plan:
                            st.session_state.show_success_message = True
                        else:
                            st.session_state.show_success_message = False
                            st.error("Failed to Generate Meal Plan. Please Try Again")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.session_state.meal_plan = None
                        st.session_state.show_success_message = False

                    finally:
                        await close_http_session()

        # ✅ SUCCESS BANNER
        if st.session_state.get("show_success_message", False):
            success_placeholder.markdown(
                """
                <div style='
                    text-align: center;
                    background-color: #d4edda;
                    padding: 10px;
                    border-radius: 5px;
                    color: #155724;
                    font-weight: bold;
                    margin-bottom: 15px;
                    margin-top: 5px;
                '>
                    Meal Plan Generated Successfully!
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            success_placeholder.empty()

        # ✅ DISPLAY PLAN
        if st.session_state.meal_plan:
            display_meal_plan(st.session_state.meal_plan)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

    finally:
        await close_http_session()

def run_app():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

# For local execution
if __name__ == "__main__":
    run_app()
