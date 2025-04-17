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
from prompts import (
    get_meal_prompt, 
    SYSTEM_PROMPT,
    DIETARY_REQUIREMENTS,
    IMPORTANT_RULES,
    AUTHENTIC_RECIPE_NAMES
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

# Set page config
st.set_page_config(
    page_title="BiteWise",
    page_icon="🍽️",
    layout="wide"
)

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
                default=[],  # Start with empty selection
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
                options=["Busy schedule (15 mins)", "Moderate schedule (30 mins)", 
                        "Busy on some days (45 mins)", "Flexible Schedule (60 mins)", 
                        "No Constraints (Any duration)"],
                index=1  # Default to "Busy on some days"
            )
            
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
            "time_constraint": str(time_constraint),  # Ensure it's a string
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

async def generate_meal(meal_type, day, prompt, cuisine="All"):
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "model": "gpt-4.1-mini", #gpt-4o-mini
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4096,
                "temperature": 0.7,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1
            }

            headers = {
                "Authorization": f"Bearer {openai.api_key}",
                "Content-Type": "application/json"
            }

            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=data,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result['choices'][0]['message']['content']

                    # 🔍 DEBUG: Save raw GPT output to file
                    with open(f"debug_day{day}_{meal_type.lower()}.txt", "w", encoding="utf-8") as f:
                        f.write(response_text)

                    # Parse and validate structure
                    recipe = parse_recipe(response_text)
                    if recipe is None or not recipe.get("name") or not recipe.get("ingredients"):
                        st.warning(f"{meal_type} for Day {day} had invalid structure. Skipping...")
                        return None

                    # Allergy/Diet/Health validation
                    if not is_recipe_safe(recipe["ingredients"], st.session_state.user_preferences):
                        st.warning(f"{meal_type} for Day {day} was rejected due to allergy/diet/health violation.")
                        return None

                    # 🔥 Strict title check for restricted diet keywords
                    if not is_title_allowed_for_diet(recipe["name"], st.session_state.user_preferences.get("diet", [])):
                        # st.warning(f"{meal_type} for Day {day} was rejected because title includes restricted food for the selected diet.")

                        print(f"[TITLE REJECTED] {meal_type} for Day {day} - Title violates diet restrictions.")

                        return None

                    # 🔥 FIX HERE: check similarity BEFORE saving embedding
                    if await is_similar_recipe(recipe["name"], recipe["ingredients"]):
                        st.warning(f"Duplicate recipe detected for {meal_type} on Day {day}, regenerating...")
                        return None

                    # ✅ NOW safe to save after confirmed not a duplicate
                    await save_recipe_embedding(recipe["name"], recipe["ingredients"])

                    return (meal_type, day, response_text)

                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")

    except Exception as e:
        st.error(f"Error generating {meal_type}: {str(e)}")
        return None

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

def is_recipe_safe(ingredients_text, user_prefs):
    ingredients_text = ingredients_text.lower()

    # 1. Allergy Check
    allergies = user_prefs.get("allergies", "")
    if "no allergies or intolerances" not in allergies.lower():
        for allergy in allergies.split(","):
            allergy = allergy.strip().lower()
            keywords = ALLERGEN_KEYWORDS.get(allergy, [])
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", ingredients_text):
                    print(f"[ALLERGY VIOLATION] Matched '{keyword}' for allergy '{allergy}'")
                    return False

    # 2. Diet Restriction Check
    diet = user_prefs.get("diet", "none")
    if diet != "None":
        if isinstance(diet, str):
            diet = eval(diet) if diet.startswith("[") else [diet]

        # Tokenize ingredients for exact match
        words = set(re.findall(r'\b\w+\b', ingredients_text))

        for d in diet:
            if d in DIET_RESTRICTIONS:
                for keyword in DIET_RESTRICTIONS[d]:
                    if keyword in words:
                        print(f"[DIET VIOLATION] Matched '{keyword}' for diet '{d}'")
                        print(f">>> Ingredients being checked: {ingredients_text}")
                        print(f">>> Forbidden keyword: {keyword} for diet: {d}")
                        return False

    # 3. Health Condition Check
    health_conditions = user_prefs.get("health_conditions", "")
    if "none" not in health_conditions.lower():
        for condition in health_conditions.split(","):
            condition = condition.strip().lower()
            keywords = HEALTH_RESTRICTIONS.get(condition, [])
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", ingredients_text):
                    print(f"[HEALTH VIOLATION] Matched '{keyword}' for condition '{condition}'")
                    return False

    return True

from openai import AsyncOpenAI  # NEW: for async calls

# Initialize OpenAI Async client globally (best practice)
client = AsyncOpenAI()

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

async def generate_meal_plan(user_prefs):
    try:
        # Initialize tracking for unique recipes and cuisine distribution
        st.session_state.generated_recipes = []
        st.session_state.meal_types_used = set()
        st.session_state.cuisines_used = set()
        st.session_state.cuisine_distribution = {}
        st.session_state.used_recipe_names = set()

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
        cooking_time = user_prefs['time_constraint']
        servings = user_prefs['serving_size']

        # Clean lists from strings
        diet_list = [d.strip().lower() for d in parse_list(diet_type)]
        allergy_list = [a.strip().lower() for a in parse_list(allergies)]
        health_list = [h.strip().lower() for h in parse_list(health_conditions)]

        formatted_prefs = {
            'num_days': num_days,
            'diet': diet_type,
            'allergies': allergies,
            'health_conditions': health_conditions,
            'meal_type': meal_type,
            'cuisine': cuisine,
            'budget': budget,
            'time_constraint': cooking_time,
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
        - Time Constraint: {cooking_time}
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

        async def limited_generate(meal_type, day, prompt, cuisine):
            async with semaphore:
                return await generate_meal(meal_type, day, prompt, cuisine)

        tasks = []
        used_cuisines = set()
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

            for meal in meal_types:
                selected_cuisine = available_cuisines[cuisine_index % len(available_cuisines)]
                cuisine_index += 1
                st.session_state.cuisine_distribution[day][meal.lower()] = selected_cuisine
                used_cuisines.add(selected_cuisine)

                authentic_recipes = AUTHENTIC_RECIPE_NAMES.get(selected_cuisine, [])
                if not authentic_recipes:
                    st.warning(f"No authentic recipes found for {selected_cuisine} cuisine.")
                    continue

                meal_recipes = [
                    r for r in authentic_recipes
                    if meal.lower() in r.lower() and is_title_allowed_for_diet(r, diet_list)
                ]

                adaptive_generation = False

                if not meal_recipes:
                    if any(
                        d in DIET_RESTRICTIONS and
                        not any(is_title_allowed_for_diet(r, [d]) for r in authentic_recipes)
                        for d in diet_list
                    ):
                        st.warning(f"⚠️ No authentic titles matched the {diet_list} diet for {selected_cuisine} {meal}. Forcing GPT to generate a compliant custom recipe.")
                        recipe_name = f"Custom {selected_cuisine} {meal} (Diet-Compliant)"
                        meal_recipes = [recipe_name]
                        adaptive_generation = True
                    else:
                        st.info(f"💡 No diet-compliant titles found for {selected_cuisine} {meal}. Entering adaptive generation mode.")
                        recipe_name = f"{selected_cuisine} {meal} - GPT Generated Fallback"
                        meal_recipes = [recipe_name]
                        adaptive_generation = True

                random.shuffle(meal_recipes)

                for recipe_name in meal_recipes:
                    if recipe_name in st.session_state.used_recipe_names:
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
                        - Include {selected_cuisine} specific ingredients
                        - Avoid mixing with other cuisines
                        - Use authentic {selected_cuisine} cooking techniques
                        - Follow {selected_cuisine} plating styles
                        - Use proper {selected_cuisine} terminology
                        - Ensure dish is recognizably {selected_cuisine}
                        - The recipe MUST explicitly mention \"{selected_cuisine}\" in its description or title

                        IMPORTANT: When generating the recipe:
                        1. GPT MUST use the exact recipe name: \"{recipe_name}\" and not change or rephrase it.
                        2. The recipe title in the output must exactly match this string, including punctuation and word order.
                        3. GPT must NOT add any extra prefix (e.g. Day 1 - Lunch -) unless explicitly instructed.
                        """ + dietary_requirements + budget_requirements + measurement_requirements,
                        available_ingredients=available_ingredients,
                        authentic_recipes=[recipe_name]
                    )

                    task = limited_generate(meal, day, prompt, selected_cuisine)
                    tasks.append(task)
                    break

        results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                meal_type, day, recipe_text = result
                st.session_state.generated_recipes.append({
                    "day": day,
                    "meal_type": meal_type,
                    "recipe": recipe_text
                })

        return st.session_state.generated_recipes

    except Exception as e:
        st.error(f"Error generating meal plan: {str(e)}")
        return []

def display_meal_plan(meal_plan):
    st.title("Your Personalized Meal Plan")
    
    # Format meal plan for display and download
    formatted_meal_plan = ""
    for meal in meal_plan:
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
    
    # Add download button with formatted text
    st.download_button(
        label="Download Meal Plan",
        data=formatted_meal_plan,
        file_name=f"meal_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    # Display the formatted meal plan
    st.markdown(formatted_meal_plan)
    
    # TEMPORARILY COMMENTED OUT: Generation time display
    # if hasattr(st.session_state, 'generation_time'):
    #     st.markdown(f"<p style='color: #666666; text-align: right; font-size: 0.9em;'>Total Generation Time: {st.session_state.generation_time:.2f} seconds</p>", unsafe_allow_html=True)

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

def main():
    try:
        user_prefs = get_user_preferences()

        if user_prefs is None:
            st.error("Failed to get user preferences. Please try again.")
            return

        st.markdown(
            """
            <style>
            div.stSpinner > div {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Generate Meal Plan"):
                with st.spinner("Generating Your Personalized Meal Plan..."):
                    try:
                        # ✅ FULL RESET (session + in-memory + optional disk)
                        st.session_state.used_recipe_names = set()
                        st.session_state.generated_recipes = []
                        st.session_state.meal_types_used = set()
                        st.session_state.cuisines_used = set()
                        st.session_state.cuisine_distribution = {}

                        recipe_index.reset()
                        recipe_names.clear()

                        # Optional disk reset for FAISS files (safe for testing)
                        if os.path.exists(INDEX_FILE):
                            os.remove(INDEX_FILE)
                        if os.path.exists(NAMES_FILE):
                            os.remove(NAMES_FILE)

                        # 🔁 Generate
                        start_time = time.time()
                        meal_plan = asyncio.run(generate_meal_plan(user_prefs))
                        st.session_state.meal_plan = meal_plan

                        if meal_plan:
                            st.markdown(
                                "<div style='text-align: center; background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724; font-weight: bold;'>"
                                "Meal Plan Generated Successfully!"
                                "</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("Failed to Generate Meal Plan. Please Try Again")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.session_state.meal_plan = None

        if st.session_state.meal_plan:
            display_meal_plan(st.session_state.meal_plan)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()