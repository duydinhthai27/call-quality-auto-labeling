# --- checkpoint_manager.py ---
from datetime import datetime
import joblib
from config import CHECKPOINT_DIR

def save_checkpoint(model, model_name, iteration):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CHECKPOINT_DIR}/{model_name}_{timestamp}_iter{iteration}.joblib"
    joblib.dump(model, filename)
