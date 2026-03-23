# import os
# from pathlib import Path

# BASE_DIR = Path(__file__).parent
# OUTPUT_DIR = BASE_DIR / "output"
# OUTPUT_DIR.mkdir(exist_ok=True)

# class Config:
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
#     MODEL_CACHE_DIR = BASE_DIR / "model_cache"
#     MAX_IMAGE_SIZE = (1024, 1024)
#     BATCH_SIZE = 16
#     DEFAULT_LLM_MODEL = "gemini-pro"
#     DEFAULT_VISION_MODEL = "facebook/deit-base-patch16-224"
#     DEFAULT_NLP_MODEL = "distilbert-base-uncased"


import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_CACHE_DIR = BASE_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # Paths
    BASE_DIR = BASE_DIR
    OUTPUT_DIR = OUTPUT_DIR
    MODEL_CACHE_DIR = MODEL_CACHE_DIR

    # Image settings
    MAX_IMAGE_SIZE = (1024, 1024)
    BATCH_SIZE = 16

    # Model defaults
    DEFAULT_LLM_MODEL = "gpt-4"
    DEFAULT_VISION_MODEL = "resnet50"
    DEFAULT_NLP_MODEL = "distilbert-base-uncased"

    # ML defaults
    N_ESTIMATORS = 150
    MAX_DEPTH = 6
    RANDOM_STATE = 42
    CV_FOLDS = 5