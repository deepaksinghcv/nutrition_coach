"""Shared configuration: model, prompt, paths, LoRA settings, Food101 nutrition map."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# The single instruction used everywhere (training + inference + eval) so the
# model always sees the same task framing.
PROMPT = """Analyze this food image and return ONLY a JSON object with this exact structure, no extra text:
{
  "food_name": "string",
  "serving_description": "string",
  "calories": number,
  "protein_g": number,
  "carbs_g": number,
  "fat_g": number
}"""

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FOOD101_DIR = DATA / "food101"                      # HuggingFace arrow dataset
N5K_METADATA = DATA / "nutrition5k" / "metadata" / "dish_metadata_cafe1.csv"
N5K_TRAIN_IMAGERY = DATA / "nutrition5k" / "imagery"
N5K_HOLDOUT_IMAGERY = DATA / "nutrition5k" / "holdout"

ADAPTERS = ROOT / "adapters"

# ---------------------------------------------------------------------------
# LoRA / training
# ---------------------------------------------------------------------------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# ---------------------------------------------------------------------------
# Food101 has no nutrition labels — map each of the 101 classes to a typical
# per-serving estimate (USDA-style averages): (calories, protein_g, carbs_g, fat_g).
# NOTE: these are class averages, not per-image measurements. The experiment
# shows this synthetic labelling actually *hurts* calorie estimation vs. the
# real measured Nutrition5k data — see README.
# ---------------------------------------------------------------------------
FOOD101_NUTRITION = {
    "apple_pie": (320, 3, 43, 14), "baby_back_ribs": (320, 25, 0, 24),
    "baklava": (330, 5, 36, 20), "beef_carpaccio": (180, 18, 3, 11),
    "beef_tartare": (200, 22, 2, 12), "beet_salad": (120, 4, 15, 6),
    "beignets": (290, 5, 38, 14), "bibimbap": (490, 22, 75, 12),
    "bread_pudding": (340, 7, 48, 14), "breakfast_burrito": (460, 22, 42, 22),
    "bruschetta": (180, 5, 24, 7), "caesar_salad": (360, 10, 20, 28),
    "cannoli": (310, 7, 35, 17), "caprese_salad": (250, 15, 8, 18),
    "carrot_cake": (420, 5, 55, 21), "ceviche": (130, 18, 8, 2),
    "cheesecake": (400, 7, 36, 26), "cheese_plate": (380, 22, 5, 31),
    "chicken_curry": (280, 25, 18, 12), "chicken_quesadilla": (490, 30, 40, 22),
    "chicken_wings": (430, 36, 15, 26), "chocolate_cake": (370, 5, 50, 18),
    "chocolate_mousse": (290, 5, 25, 20), "churros": (330, 4, 40, 17),
    "clam_chowder": (200, 8, 22, 9), "club_sandwich": (540, 32, 42, 26),
    "crab_cakes": (290, 20, 15, 17), "creme_brulee": (330, 5, 33, 21),
    "croque_madame": (520, 30, 35, 28), "cup_cakes": (300, 3, 42, 14),
    "deviled_eggs": (140, 9, 2, 11), "donuts": (300, 4, 38, 15),
    "dumplings": (330, 14, 42, 12), "edamame": (120, 11, 9, 5),
    "eggs_benedict": (400, 20, 24, 25), "escargots": (230, 18, 5, 16),
    "falafel": (330, 13, 35, 17), "filet_mignon": (400, 38, 0, 27),
    "fish_and_chips": (600, 28, 65, 26), "foie_gras": (460, 10, 5, 44),
    "french_fries": (365, 4, 48, 17), "french_onion_soup": (240, 12, 28, 9),
    "french_toast": (280, 10, 36, 11), "fried_calamari": (290, 16, 22, 16),
    "fried_rice": (360, 11, 58, 10), "frozen_yogurt": (200, 5, 40, 3),
    "garlic_bread": (190, 4, 24, 9), "gnocchi": (250, 7, 48, 4),
    "greek_salad": (180, 5, 12, 13), "grilled_cheese_sandwich": (440, 18, 40, 24),
    "grilled_salmon": (280, 38, 0, 13), "guacamole": (150, 2, 8, 14),
    "gyoza": (260, 12, 30, 10), "hamburger": (540, 28, 42, 28),
    "hot_and_sour_soup": (130, 9, 15, 4), "hot_dog": (290, 12, 24, 16),
    "huevos_rancheros": (350, 18, 30, 18), "hummus": (170, 8, 14, 10),
    "ice_cream": (270, 4, 33, 14), "lasagna": (340, 18, 32, 15),
    "lobster_bisque": (280, 14, 18, 18), "lobster_roll_sandwich": (430, 28, 30, 22),
    "macaroni_and_cheese": (310, 12, 42, 11), "macarons": (210, 3, 32, 8),
    "miso_soup": (40, 3, 5, 1), "mussels": (200, 27, 9, 5),
    "nachos": (480, 14, 52, 25), "omelette": (220, 16, 2, 17),
    "onion_rings": (380, 5, 44, 20), "oysters": (100, 12, 6, 3),
    "pad_thai": (430, 22, 52, 15), "paella": (400, 24, 45, 13),
    "pancakes": (350, 9, 54, 11), "panna_cotta": (240, 4, 30, 12),
    "peking_duck": (400, 28, 20, 24), "pho": (350, 22, 45, 8),
    "pizza": (285, 12, 36, 10), "pork_chop": (320, 32, 0, 21),
    "poutine": (530, 14, 60, 28), "prime_rib": (500, 40, 0, 38),
    "pulled_pork_sandwich": (490, 32, 42, 20), "ramen": (430, 22, 55, 13),
    "ravioli": (310, 14, 40, 11), "red_velvet_cake": (380, 5, 52, 18),
    "risotto": (330, 10, 52, 9), "samosa": (260, 5, 30, 14),
    "sashimi": (130, 24, 0, 3), "scallops": (180, 22, 8, 5),
    "seaweed_salad": (90, 2, 10, 5), "shrimp_and_grits": (400, 26, 38, 16),
    "spaghetti_bolognese": (420, 24, 48, 14), "spaghetti_carbonara": (440, 22, 48, 18),
    "spring_rolls": (220, 7, 26, 10), "steak": (450, 42, 0, 30),
    "strawberry_shortcake": (320, 5, 44, 14), "sushi": (200, 10, 32, 3),
    "tacos": (370, 20, 30, 18), "takoyaki": (230, 10, 28, 9),
    "tiramisu": (380, 7, 38, 22), "tuna_tartare": (170, 20, 4, 8),
    "waffles": (360, 8, 48, 15),
}
