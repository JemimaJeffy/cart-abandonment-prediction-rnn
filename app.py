import streamlit as st
import random
from datetime import datetime, timedelta
import uuid
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
import joblib


# --- START: Production Model and Artifact Loading ---
@st.cache_resource
def load_production_model_artifacts():
    """
    Loads the production Keras model, scalers, and config from files.
    """
    # Define file paths
    config_path = 'retail_rocket_relational_model_config.json'
    model_path = 'retail_rocket_relational_rnn_model.keras'
    scalers_path = 'retail_rocket_relational_scalers.joblib'

    # Load the configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load the Keras model
    model = load_model(model_path)

    # Load the scalers (assumes it's a dictionary of scalers)
    scalers = joblib.load(scalers_path)

    return model, scalers, config


# --- END: Production Model and Artifact Loading ---


# --- Configuration will be loaded from the JSON file ---
# Constants needed for functions outside the main prediction flow
CLOTHING_DATA = {
    "Tops": [
        {"name": "T-Shirt", "image_filename": "tshirt.png", "price": 599.00},
        {"name": "Blouse", "image_filename": "blouse.png", "price": 1299.00},
        {"name": "Cropped Hoodie", "image_filename": "hoodie.jpg", "price": 1499.00},
        {"name": "Shirt", "image_filename": "shirt.png", "price": 1199.00},
        {"name": "Top", "image_filename": "top.png", "price": 799.00},
        {"name": "Oversized T-Shirt", "image_filename": "oversizedtshirt.png", "price": 699.00},
    ],
    "Jeans": [
        {"name": "Skinny Jeans", "image_filename": "skinnyjeans.png", "price": 1899.00},
        {"name": "Straight Jeans", "image_filename": "straightjeans.png", "price": 1999.00},
        {"name": "Vintage Jeans", "image_filename": "vintagejeans.png", "price": 2299.00},
        {"name": "Black Denim", "image_filename": "blackdenim.png", "price": 2199.00},
        {"name": "Distressed Jeans", "image_filename": "distressedjeans.png", "price": 2499.00},
        {"name": "Slim Fit Jeans", "image_filename": "slimfitjeans.png", "price": 1799.00},
    ],
    "Shoes": [
        {"name": "Sneakers", "image_filename": "sneakers.png", "price": 2499.00},
        {"name": "Heels", "image_filename": "heels.png", "price": 1899.00},
        {"name": "Boots", "image_filename": "boots.png", "price": 3499.00},
        {"name": "Loafers", "image_filename": "loafers.png", "price": 1599.00},
        {"name": "Wedges", "image_filename": "wedges.png", "price": 1799.00},
        {"name": "Trainers", "image_filename": "balletpumps.png", "price": 2999.00},
    ],
    "Accessories": [
        {"name": "Necklace", "image_filename": "necklace.png", "price": 799.00},
        {"name": "Bag", "image_filename": "bag.png", "price": 2199.00},
        {"name": "Scarf", "image_filename": "scarf.png", "price": 499.00},
        {"name": "Sunglasses", "image_filename": "sunglasses.png", "price": 1299.00},
        {"name": "Hat", "image_filename": "hat.png", "price": 899.00},
        {"name": "Bracelet", "image_filename": "bracelet.png", "price": 399.00},
    ]
}

REMOVE_ICON = "🗑️"
UNKNOWN_ITEM_ID_PLACEHOLDER = 0
UNKNOWN_CATEGORY_ID_PLACEHOLDER = 0
MAX_LOOKBACK_FOR_VIEW_ADD_RELATIONAL = 3


def generate_clothing_products():
    products = []
    for category_name, items in CLOTHING_DATA.items():
        for item_details in items:
            image_path = os.path.join(category_name.lower(), item_details["image_filename"])
            products.append({
                "id": str(uuid.uuid4()),
                "name": item_details["name"],
                "category": category_name,
                "price": item_details["price"],
                "image_path": image_path,
                "available": random.choice([True, True, True, False, False]),
                "description": f"A stylish {item_details['name'].lower()} from our {category_name} collection."
            })
    return products


def initialize_session_state():
    if "rnn_model" not in st.session_state:
        # Call the new function to load production artifacts
        model, scalers, config = load_production_model_artifacts()
        st.session_state.rnn_model = model
        st.session_state.rnn_scalers = scalers
        st.session_state.rnn_model_config = config
        st.session_state.item_uuid_to_int_id_map = {}
        st.session_state.next_item_int_id = 1
        st.session_state.category_name_to_int_id_map = {name: i for i, name in enumerate(CLOTHING_DATA.keys(), 1)}

    if "products" not in st.session_state:
        st.session_state.products = generate_clothing_products()
    if "categories" not in st.session_state:
        st.session_state.categories = list(CLOTHING_DATA.keys())
    if "cart" not in st.session_state:
        st.session_state.cart = []
    if "action_log" not in st.session_state:
        st.session_state.action_log = []
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = "All"
    if "prediction" not in st.session_state:
        st.session_state.prediction = "N/A - No actions yet."
    if "last_logged_category_view" not in st.session_state:
        st.session_state.last_logged_category_view = None


def log_action(action_type, item_id=None, category_id=None, details=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"timestamp": timestamp, "action": action_type}
    if item_id: log_entry["item_id"] = str(item_id)
    if category_id: log_entry["category_id"] = category_id
    if details: log_entry["details"] = details
    st.session_state.action_log.append(log_entry)
    MAX_LOG_SIZE = 50
    if len(st.session_state.action_log) > MAX_LOG_SIZE:
        st.session_state.action_log = st.session_state.action_log[-MAX_LOG_SIZE:]

    update_rnn_prediction()


def get_product_by_id(product_id):
    for p in st.session_state.products:
        if p["id"] == product_id: return p
    return None


def get_or_create_item_int_id(item_uuid_str):
    if item_uuid_str is None:
        return UNKNOWN_ITEM_ID_PLACEHOLDER
    if item_uuid_str not in st.session_state.item_uuid_to_int_id_map:
        st.session_state.item_uuid_to_int_id_map[item_uuid_str] = st.session_state.next_item_int_id
        st.session_state.next_item_int_id += 1
    return st.session_state.item_uuid_to_int_id_map[item_uuid_str]


def get_category_int_id(category_name_str):
    if category_name_str is None:
        return UNKNOWN_CATEGORY_ID_PLACEHOLDER
    return st.session_state.category_name_to_int_id_map.get(category_name_str, UNKNOWN_CATEGORY_ID_PLACEHOLDER)


def generate_relational_features_for_streamlit_session(session_event_dicts):
    config = st.session_state.rnn_model_config
    RELATIONAL_NUM_FEATURES = config['RELATIONAL_NUM_FEATURES']
    REL_FT_INDICES = config['REL_FT_INDICES']
    EVENT_MAP_MODEL = config['EVENT_MAP']

    processed_event_features = []
    last_event_time_dt = None
    last_interacted_item_int_id = None
    last_interacted_category_int_id = None
    current_cart_item_int_ids = set()
    current_cart_category_int_ids = {}

    for i, event_dict in enumerate(session_event_dicts):
        feature_vector = np.zeros(RELATIONAL_NUM_FEATURES, dtype=np.float32)

        if event_dict['event_code'] == EVENT_MAP_MODEL['view']:
            feature_vector[REL_FT_INDICES['F_EVENT_CODE_VIEW']] = 1.0
        elif event_dict['event_code'] == EVENT_MAP_MODEL['addtocart']:
            feature_vector[REL_FT_INDICES['F_EVENT_CODE_ADDTOCART']] = 1.0

        if last_event_time_dt:
            time_delta_seconds = (event_dict['timestamp_dt'] - last_event_time_dt).total_seconds()
            feature_vector[REL_FT_INDICES['F_TIME_SINCE_LAST_EVENT']] = max(0, time_delta_seconds)
        else:
            feature_vector[REL_FT_INDICES['F_TIME_SINCE_LAST_EVENT']] = 0.0

        feature_vector[REL_FT_INDICES['F_ITEM_AVAILABLE_AT_EVENT']] = float(event_dict.get('available_flag', 0.0))

        feature_vector[REL_FT_INDICES['F_ITEMS_IN_CART_SO_FAR']] = float(len(current_cart_item_int_ids))
        distinct_cats_in_cart = set(current_cart_category_int_ids.values())
        feature_vector[REL_FT_INDICES['F_DISTINCT_CATEGORIES_IN_CART_SO_FAR']] = float(len(distinct_cats_in_cart))

        current_event_item_int_id = event_dict.get('item_int_id')
        current_event_category_int_id = event_dict.get('category_int_id')

        if event_dict['event_code'] == EVENT_MAP_MODEL[
            'addtocart'] and current_event_item_int_id != UNKNOWN_ITEM_ID_PLACEHOLDER:
            for k in range(1, MAX_LOOKBACK_FOR_VIEW_ADD_RELATIONAL + 1):
                if i - k >= 0:
                    prev_event_dict = session_event_dicts[i - k]
                    if prev_event_dict['event_code'] == EVENT_MAP_MODEL['view'] and prev_event_dict.get(
                            'item_int_id') == current_event_item_int_id:
                        feature_vector[REL_FT_INDICES['F_VIEWED_THEN_ADDED_SAME_ITEM']] = 1.0
                        break

        if current_event_item_int_id != UNKNOWN_ITEM_ID_PLACEHOLDER and last_interacted_item_int_id is not None:
            if current_event_item_int_id == last_interacted_item_int_id:
                feature_vector[REL_FT_INDICES['F_SAME_ITEM_AS_LAST_ITEM_INTERACTION']] = 1.0
            elif current_event_category_int_id != UNKNOWN_CATEGORY_ID_PLACEHOLDER and last_interacted_category_int_id != UNKNOWN_CATEGORY_ID_PLACEHOLDER and current_event_category_int_id == last_interacted_category_int_id:
                feature_vector[REL_FT_INDICES['F_SAME_CATEGORY_AS_LAST_ITEM_INTERACTION']] = 1.0
        elif event_dict['event_code'] == EVENT_MAP_MODEL[
            'view'] and current_event_item_int_id == UNKNOWN_ITEM_ID_PLACEHOLDER and current_event_category_int_id != UNKNOWN_CATEGORY_ID_PLACEHOLDER and last_interacted_category_int_id != UNKNOWN_CATEGORY_ID_PLACEHOLDER and current_event_category_int_id == last_interacted_category_int_id:
            feature_vector[REL_FT_INDICES['F_SAME_CATEGORY_AS_LAST_ITEM_INTERACTION']] = 1.0

        processed_event_features.append(feature_vector)

        last_event_time_dt = event_dict['timestamp_dt']
        if current_event_item_int_id != UNKNOWN_ITEM_ID_PLACEHOLDER:
            last_interacted_item_int_id = current_event_item_int_id
            last_interacted_category_int_id = current_event_category_int_id
        elif event_dict['event_code'] == EVENT_MAP_MODEL[
            'view'] and current_event_item_int_id == UNKNOWN_ITEM_ID_PLACEHOLDER:
            last_interacted_item_int_id = UNKNOWN_ITEM_ID_PLACEHOLDER
            last_interacted_category_int_id = current_event_category_int_id

        if event_dict['event_code'] == EVENT_MAP_MODEL[
            'addtocart'] and current_event_item_int_id != UNKNOWN_ITEM_ID_PLACEHOLDER:
            current_cart_item_int_ids.add(current_event_item_int_id)
            if current_event_category_int_id != UNKNOWN_CATEGORY_ID_PLACEHOLDER:
                current_cart_category_int_ids[current_event_item_int_id] = current_event_category_int_id

    return np.array(processed_event_features, dtype=np.float32)


def extract_rnn_input_sequence_from_log():
    config = st.session_state.rnn_model_config
    EVENT_MAP_MODEL = config['EVENT_MAP']
    MAX_SEQ_LEN = config['MAX_SEQ_LEN']

    event_dicts_for_rnn = []
    log_to_process = list(st.session_state.action_log)

    for log_entry in log_to_process:
        action_type = log_entry['action']
        event_dict_for_feature_extractor = {}
        product_details = None

        if log_entry.get('item_id'):
            product_details = get_product_by_id(log_entry['item_id'])

        if action_type == 'view_product_detail' and product_details:
            event_dict_for_feature_extractor['event_code'] = EVENT_MAP_MODEL['view']
            event_dict_for_feature_extractor['item_int_id'] = get_or_create_item_int_id(product_details['id'])
            event_dict_for_feature_extractor['category_int_id'] = get_category_int_id(product_details['category'])
            event_dict_for_feature_extractor['available_flag'] = 1.0 if product_details['available'] else 0.0
        elif action_type == 'add_to_cart' and product_details:
            event_dict_for_feature_extractor['event_code'] = EVENT_MAP_MODEL['addtocart']
            event_dict_for_feature_extractor['item_int_id'] = get_or_create_item_int_id(product_details['id'])
            event_dict_for_feature_extractor['category_int_id'] = get_category_int_id(product_details['category'])
            event_dict_for_feature_extractor['available_flag'] = 1.0 if product_details['available'] else 0.0
        elif action_type == 'view_category':
            event_dict_for_feature_extractor['event_code'] = EVENT_MAP_MODEL['view']
            event_dict_for_feature_extractor['item_int_id'] = UNKNOWN_ITEM_ID_PLACEHOLDER
            event_dict_for_feature_extractor['category_int_id'] = get_category_int_id(log_entry.get('category_id'))
            event_dict_for_feature_extractor['available_flag'] = 1.0
        else:
            continue

        event_dict_for_feature_extractor['timestamp_dt'] = datetime.strptime(log_entry['timestamp'],
                                                                             "%Y-%m-%d %H:%M:%S")
        event_dicts_for_rnn.append(event_dict_for_feature_extractor)

    if not event_dicts_for_rnn:
        return np.array([])

    relevant_event_dicts = event_dicts_for_rnn[-MAX_SEQ_LEN:]
    relational_feature_sequence = generate_relational_features_for_streamlit_session(relevant_event_dicts)
    return relational_feature_sequence


def update_rnn_prediction():
    config = st.session_state.rnn_model_config
    MAX_SEQ_LEN = config['MAX_SEQ_LEN']
    RELATIONAL_NUM_FEATURES = config['RELATIONAL_NUM_FEATURES']
    REL_FT_NUMERICAL_INDICES = config['REL_FT_NUMERICAL_INDICES']
    REL_FT_INDICES = config['REL_FT_INDICES']

    feature_sequence = extract_rnn_input_sequence_from_log()

    if feature_sequence.ndim == 0 or feature_sequence.shape[0] == 0:
        st.session_state.prediction = "N/A - Insufficient actions for prediction."
        return

    if feature_sequence.ndim != 2 or feature_sequence.shape[1] != RELATIONAL_NUM_FEATURES:
        st.session_state.prediction = "Error: Feature sequence has incorrect dimensions."
        return

    padded_sequence = pad_sequences([feature_sequence], maxlen=MAX_SEQ_LEN, padding='post', truncating='post',
                                    dtype='float32', value=0.0)

    # --- Start of Corrected Section ---
    # Replace the vectorized scaling with a safer loop-based approach.
    for feature_idx in REL_FT_NUMERICAL_INDICES:
        if feature_idx in st.session_state.rnn_scalers:
            scaler = st.session_state.rnn_scalers[feature_idx]
            # Loop through each timestep in the sequence
            for t in range(padded_sequence.shape[1]):
                # Check if this timestep is a real event (not padding)
                is_real_event = np.any(padded_sequence[0, t, [REL_FT_INDICES['F_EVENT_CODE_VIEW'], REL_FT_INDICES['F_EVENT_CODE_ADDTOCART']]] != 0.0)
                if is_real_event:
                    # Extract the single value, reshape for the scaler, transform, and update in place
                    value_to_scale = padded_sequence[0, t, feature_idx].reshape(1, -1)
                    scaled_value = scaler.transform(value_to_scale)
                    padded_sequence[0, t, feature_idx] = scaled_value[0, 0]
    # --- End of Corrected Section ---

    try:
        raw_prediction = st.session_state.rnn_model.predict(padded_sequence)
        abandonment_probability = raw_prediction[0][0]
    except Exception as e:
        st.error(f"RNN Prediction Error: {e}")
        st.session_state.prediction = "Error during prediction."
        return

    if abandonment_probability > 0.75:
        pred_text = f"High Risk of Abandonment (Prob: {abandonment_probability:.2f})"
    elif abandonment_probability > 0.5:
        pred_text = f"Medium-High Risk (Prob: {abandonment_probability:.2f})"
    elif abandonment_probability > 0.25:
        pred_text = f"Medium-Low Risk (Prob: {abandonment_probability:.2f})"
    else:
        pred_text = f"Low Risk of Abandonment (Prob: {abandonment_probability:.2f})"

    if not st.session_state.cart:
        pred_text = "N/A - Cart is empty."

    st.session_state.prediction = pred_text


# --- All UI functions below this line remain the same ---
def display_light_theme_css():
    st.markdown(f"""
    <style>
    /* 1. Main App Background & General Text */
    .stApp {{
        background-color: #FFFFFF !important;
    }}
    body, .stApp, .stMarkdown, .stSubheader, .stHeader, .stTitle, .stCaption, label, th, td, p, li {{
        color: #262730 !important; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #1E1F24 !important; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }}

    /* Top header/toolbar area - Change from black to white */
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    .stApp > header,
    section[data-testid="stHeader"] {{
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #ECECEC !important;
    }}

    /* --- ALL BUTTONS WHITE STYLING --- */

    /* Universal button styling - ensures ALL buttons are white */
    div.stButton > button,
    button,
    .stButton button {{
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #ECECEC !important;
        border-radius: 0.3rem !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem !important;
        font-weight: 500 !important;
        text-align: center !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        transition: color 0.15s ease-in-out, background-color 0.15s ease-in-out, border-color 0.15s ease-in-out;
        cursor: pointer !important;
    }}

    /* Hover state for all buttons */
    div.stButton > button:hover,
    button:hover,
    .stButton button:hover {{
        background-color: #F8F9FA !important;
        color: #333333 !important;
        border-color: #ECECEC !important;
    }}

    /* Focus state for all buttons */
    div.stButton > button:focus,
    button:focus,
    .stButton button:focus {{
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1) !important;
    }}

    /* 2. Category Navigation Buttons (Top Bar) - Remove cards/borders */
    div.stButton > button:not([kind="primary"]):not([kind="secondary"]) {{ 
        background-color: transparent !important; 
        color: #333333 !important;          
        border: none !important;              
        padding: 0.5rem 0.75rem !important;    
        margin: 0 0.05rem !important;          
        border-radius: 0 !important;    
        font-weight: 500 !important;          
        text-align: center !important;
        box-shadow: none !important;
        transition: color 0.15s ease-in-out, background-color 0.15s ease-in-out;
    }}
    div.stButton > button:not([kind="primary"]):not([kind="secondary"]):hover {{
        color: #333333 !important;            
        background-color: transparent !important; 
        border-color: transparent !important;
    }}
    div.stButton > button:not([kind="primary"]):not([kind="secondary"]):focus {{
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1) !important; 
    }}
    div.stButton > button:not([kind="primary"]):not([kind="secondary"]) strong {{ 
        color: #000000 !important; 
        font-weight: bold !important;
    }}

    /* 3. Primary Action Buttons (Add to Cart) - Black button with white text */
    div.stButton > button[kind="primary"] {{
        background-color: #000000 !important;    
        color: #FFFFFF !important;               
        border: 1px solid #000000 !important;  
        border-radius: 0.3rem !important;
        padding: 0.5rem 1rem !important;      
        margin-top: 0.5rem !important;        
        font-weight: bold !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important; 
        transition: background-color 0.15s ease-in-out, color 0.15s ease-in-out, border-color 0.15s ease-in-out;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: #333333 !important; 
        color: #FFFFFF !important;            
        border-color: #333333 !important;     
    }}
    /* Force white text on primary buttons */
    div.stButton > button[kind="primary"] * {{
        color: #FFFFFF !important;
    }}
     div.stButton > button[kind="primary"]:focus {{
        box-shadow: 0 0 0 0.2rem rgba(0, 0, 0, 0.1) !important; 
    }}

    /* 4. Secondary Action Buttons (View Details) - Remove hover effects */
    div.stButton > button[kind="secondary"] {{
        background-color: #FFFFFF !important;   
        color: #4A4A4A !important;             
        border: 1px solid #ECECEC !important;   
        border-radius: 0.3rem !important;
        padding: 0.45rem 1rem !important;       
        margin-top: 0.5rem !important;
        font-weight: 500 !important;           
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        line-height: 1.5;
        transition: none !important;
    }}
    div.stButton > button[kind="secondary"]:hover {{
        background-color: #FFFFFF !important; 
        border-color: #ECECEC !important;
        color: #4A4A4A !important; 
        transform: none !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }}
    div.stButton > button[kind="secondary"]:disabled,
    div.stButton > button[kind="secondary"][disabled] {{ 
        background-color: #FFFFFF !important; 
        color: #B0B0B0 !important; 
        border-color: #E0E0E0 !important;
        cursor: not-allowed !important;
    }}

    /* Icon button specific styling */
    div.stButton > button[kind="secondary"].remove-button-icon {{
        background-color: #FFFFFF !important;
        padding: 0.3rem 0.5rem !important; 
        font-size: 1.1em !important;
        line-height: 1 !important;
        border: 1px solid #ECECEC !important;
    }}

    /* 5. Product Cards - One card per product (containing image, name, price, buttons) */
    div.stColumn[data-testid="column"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #ECECEC !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }}

    /* Remove cards from images themselves */
    div.stColumn img,
    div[data-testid="stImage"] {{
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* Remove cards from main sections */
    div[data-testid="stVerticalBlock"],
    div[data-testid="block-container"],
    .main .block-container {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }}

    /* 6. Cart Card - Much narrower sidebar with increased spacing */
    div[data-testid="stSidebar"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #ECECEC !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
        padding: 0.75rem !important;
        margin: 0.5rem !important;
        width: 120px !important;
        max-width: 120px !important;
    }}

    /* Make sidebar container much narrower */
    .stSidebar {{
        width: 140px !important;
        max-width: 140px !important;
        min-width: 140px !important;
    }}

    /* Alternative selector for sidebar */
    section[data-testid="stSidebar"] {{
        width: 140px !important;
        max-width: 140px !important;
        min-width: 140px !important;
    }}

    /* INCREASED SPACING: Create much more space between main content and cart */
    .main .block-container {{
        margin-right: 200px !important;
        padding-right: 8rem !important;
    }}

    /* Main content area adjustment for increased spacing */
    div[data-testid="stMain"] {{
        margin-right: 400px !important;
        padding-right: 8rem !important;
    }}

    /* Additional spacing for main content columns */
    div[data-testid="column"]:first-child {{
        padding-right: 4rem !important;
        margin-right: 2rem !important;
    }}

    /* Create visual separation with a subtle border */
    div[data-testid="column"]:first-child::after {{
        content: '';
        position: absolute;
        right: -2rem;
        top: 0;
        bottom: 0;
        width: 1px;
        background-color: #ECECEC;
        opacity: 0.5;
    }}

    /* 7. Message boxes for light theme */
    div[data-testid="stInfo"] {{ background-color: #E0F7FA !important; border-left: 4px solid #0288D1 !important; color: #01579B !important; border-radius: 0.25rem; }}
    div[data-testid="stSuccess"] {{ background-color: #E8F5E9 !important; border-left: 4px solid #388E3C !important; color: #1B5E20 !important; border-radius: 0.25rem;}}
    div[data-testid="stWarning"] {{ background-color: #FFF8E1 !important; border-left: 4px solid #FFA000 !important; color: #856404 !important; border-radius: 0.25rem;}}
    div[data-testid="stWarning"] p {{ color: #856404 !important; }}
    div[data-testid="stError"] {{ background-color: #FFEBEE !important; border-left: 4px solid #D32F2F !important; color: #B71C1C !important; border-radius: 0.25rem;}}
    div[data-testid="stError"] p {{ color: #B71C1C !important; }}

    /* 8. REDUCED SPACING - Cart Button Improvements */
    /* Reduce spacing between checkout and reset buttons */
    .cart-buttons-container {{
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }}

    /* Ensure minimal spacing between cart buttons */
    .cart-buttons-container .stButton {{
        margin-bottom: 0.25rem !important;
        margin-top: 0.25rem !important;
    }}

    /* 9. IMPROVED TOAST NOTIFICATIONS */
    /* Make toast notifications more readable with better contrast */
    .stToast {{
        background-color: #2E2E2E !important;
        color: #FFFFFF !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        border: 1px solid #4A4A4A !important;
    }}

    .stToast div, .stToast p, .stToast span {{
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
    }}

    /* Success toast styling */
    .stToast[data-baseweb="notification"] {{
        background-color: #1B5E20 !important;
        border-color: #388E3C !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def set_selected_category(category_name):
    if st.session_state.selected_category != category_name:
        st.session_state.selected_category = category_name
        if st.session_state.last_logged_category_view != category_name:
            log_action("view_category", category_id=category_name)
            st.session_state.last_logged_category_view = category_name


def display_top_category_bar():
    categories_for_bar = ["All"] + st.session_state.categories
    cols = st.columns(len(categories_for_bar))
    for i, category_name in enumerate(categories_for_bar):
        with cols[i]:
            is_active = (st.session_state.selected_category == category_name)
            label_to_display = f"**{category_name}**" if is_active else category_name
            st.button(
                label_to_display,
                key=f"top_cat_btn_{category_name.replace(' ', '_').replace('&', 'and')}",
                use_container_width=True,
                on_click=set_selected_category,
                args=(category_name,)
            )


def display_cart_area():
    st.header("Your Cart")
    if not st.session_state.cart:
        st.info("Your cart is empty. Start shopping!")
    else:
        total_price = 0
        cart_items_with_indices = list(enumerate(st.session_state.cart))
        for i, item_in_cart in reversed(cart_items_with_indices):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(f"**{item_in_cart['name']}**")
                st.caption(f"Rs. {item_in_cart['price']:.2f}")
            with col2:
                if col2.button(REMOVE_ICON, key=f"remove_cart_{item_in_cart['id']}_{i}",
                               help=f"Remove {item_in_cart['name']}", type="secondary"):
                    original_index = len(st.session_state.cart) - 1 - i
                    removed_item = st.session_state.cart.pop(original_index)
                    log_action("remove_from_cart", item_id=removed_item['id'], details={"name": removed_item['name']})
                    st.rerun()
            st.markdown("""<div style="margin-bottom: 0.5rem;"></div>""", unsafe_allow_html=True)
            total_price += item_in_cart['price']
        st.subheader(f"Total: Rs. {total_price:.2f}")

        st.markdown('<div class="cart-buttons-container">', unsafe_allow_html=True)
        if st.button("Proceed to Checkout", type="primary", use_container_width=True, key="checkout_action_btn"):
            log_action("checkout_start")

        if st.button("Reset Session", key="clear_session_cart_area", use_container_width=True, type="secondary"):
            keys_to_delete = list(st.session_state.keys())
            for key in keys_to_delete: del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")


def _add_to_cart_on_click(prod_to_add):
    st.session_state.cart.append(prod_to_add)
    log_action("add_to_cart", item_id=prod_to_add['id'],
               details={"name": prod_to_add['name'], "price": prod_to_add['price'],
                        "category": prod_to_add['category']})
    st.toast(f"{prod_to_add['name']} added to cart! (Rs. {prod_to_add['price']:.2f})")


def display_products_grid(products_to_display):
    if not products_to_display:
        st.info(f"No products to display for this selection.")
        return

    cols_per_row = 3
    current_display_category_key = st.session_state.selected_category.lower().replace(' ', '_').replace('&', 'and')

    for i in range(0, len(products_to_display), cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")
        for j in range(cols_per_row):
            if i + j < len(products_to_display):
                product = products_to_display[i + j]
                with cols[j]:
                    with st.container():
                        try:
                            st.image(product["image_path"], use_container_width=True)
                        except FileNotFoundError:
                            st.warning(f"Image not found: {product['image_path']}")
                            st.image(f"https://via.placeholder.com/200x250.png?text=Img+Missing",
                                     use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading: {product['image_path']}")
                            st.image(f"https://via.placeholder.com/200x250.png?text=Img+Error",
                                     use_container_width=True)

                        st.subheader(product["name"])
                        st.write(f"**Price:** Rs. {product['price']:.2f}")
                        button_key_prefix = f"{current_display_category_key}_{product['id']}_action"

                        if st.button("View Details", key=f"view_{button_key_prefix}", help=product['description'],
                                     use_container_width=True, type="secondary"):
                            log_action("view_product_detail", item_id=product['id'],
                                       details={"name": product['name'], "category": product['category']})
                            st.toast(f"Viewing: {product['name']} - {product['description']}")

                        if product["available"]:
                            st.button("Add to Cart", key=f"add_{button_key_prefix}", type="primary",
                                      use_container_width=True, on_click=_add_to_cart_on_click, args=(product,))
                        else:
                            st.button("Out of Stock", key=f"add_{button_key_prefix}", type="secondary",
                                      use_container_width=True, disabled=True)
                    st.markdown("""<div style="height: 1rem;"></div>""", unsafe_allow_html=True)


def display_prediction_and_log_main_area():
    st.markdown("## Session Analysis")
    pred_text = st.session_state.prediction
    if "High Risk" in pred_text:
        st.error(f"**Cart Outlook:** {pred_text}")
    elif "Medium-High" in pred_text or "Medium-Low" in pred_text:
        st.warning(f"**Cart Outlook:** {pred_text}")
    elif "Low Risk" in pred_text:
        st.success(f"**Cart Outlook:** {pred_text}")
    else:
        st.info(f"**Cart Outlook:** {pred_text}")

    with st.expander("View Recent Activity (Last 20)", expanded=False):
        if st.session_state.action_log:
            for entry in reversed(st.session_state.action_log):
                details_str = ""
                if entry.get('details'):
                    if isinstance(entry['details'], dict):
                        details_str = f" ({', '.join(f'{k}: {v}' for k, v in entry['details'].items())})"
                    else:
                        details_str = f" ({entry['details']})"
                item_str = f" (Item: {str(entry.get('item_id', 'N/A'))[:8]}...)" if entry.get('item_id') else ""
                cat_str = f" (Category: {entry.get('category_id', 'N/A')})" if entry.get('category_id') else ""
                st.caption(f"{entry['timestamp']} - **{entry['action']}**{item_str}{cat_str}{details_str}")
        else:
            st.caption("No actions logged yet.")


def main():
    st.set_page_config(page_title="Cart Abandonment Prediction", layout="wide", initial_sidebar_state="collapsed")
    initialize_session_state()
    display_light_theme_css()

    main_content_col, cart_col = st.columns([0.75, 0.25], gap="large")

    with main_content_col:
        st.markdown("<div class='top-content-wrapper'>", unsafe_allow_html=True)
        st.title("Cart Abandonment Prediction")
        st.caption("Your Destination for Stylish Apparel")
        display_top_category_bar()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div style="margin-top: 0.5rem; margin-bottom: 1rem;"></div>""", unsafe_allow_html=True)

        active_category_display = st.session_state.selected_category
        st.header(f"{active_category_display} Collection")

        if active_category_display == "All":
            products_to_show = st.session_state.products
        else:
            products_to_show = [p for p in st.session_state.products if p["category"] == active_category_display]

        display_products_grid(products_to_show)

        st.markdown("---")
        display_prediction_and_log_main_area()

    with cart_col:
        display_cart_area()


if __name__ == "__main__":
    main()
