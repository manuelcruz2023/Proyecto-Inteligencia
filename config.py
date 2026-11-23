"""Configuración central del proyecto. Todos los scripts y la app leen de aquí."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------------- Rutas -----------------------------
RAW_DIR = ROOT / "data" / "raw" / "UTA-RLDD"
PROCESSED_DIR = ROOT / "data" / "processed"
FACES_DIR = PROCESSED_DIR / "faces"                 # un .npy por video
METADATA_CSV = PROCESSED_DIR / "metadata.csv"
SPLITS_JSON = PROCESSED_DIR / "splits.json"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "drowsiness_cnn_lstm.keras"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

for _d in (PROCESSED_DIR, FACES_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------- Clases -----------------------------
CLASS_NAMES = ["Alert", "Low Vigilance", "Drowsy"]
# En UTA-RLDD el nombre del video indica la clase: 0 = alert, 5 = low vigilance, 10 = drowsy
LABEL_MAP = {0: 0, 5: 1, 10: 2}

# ----------------------------- Preprocesamiento -----------------------------
IMG_SIZE = 96              # MobileNetV2 admite 96, 128, 160, 192, 224
SEQ_LEN = 16               # frames por secuencia
SAMPLE_FPS = 2.0           # 16 frames / 2 fps = ventana de 8 s
MAX_SECONDS_PER_VIDEO = 300  # 0 = usar el video completo (más disco y más tiempo)
SEQ_STRIDE_TRAIN = 4       # solape alto en entrenamiento (más muestras)
SEQ_STRIDE_EVAL = 8        # solape menor en val/test

# ----------------------------- Entrenamiento -----------------------------
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
BATCH_SIZE = 16
EPOCHS_FROZEN = 8
EPOCHS_FINETUNE = 12
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_AT = 100         # capas del backbone (índice) a partir de las cuales se descongela

# ----------------------------- Servidor -----------------------------
APP_HOST = "127.0.0.1"
APP_PORT = 5000