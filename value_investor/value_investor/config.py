from pathlib import Path

# important dirs
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "2020Q1Q2Q3Q4-2021Q1.xlsx"
EXCHANGE_RATES_FILE = DATA_DIR / "exchange_rates_2020_2021Q1.csv"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

# LSTM parameters
WINDOW_SIZE = 20        # Number of time steps to look back
HORIZON = 1          # Number of time steps to predict
BATCH_SIZE = 64        # Batch size for training
NUM_UNITS = 32        # Number of LSTM units
EPOCHS = 100          # Number of training epochs
DROPOUT_PROP = 0.2    # Dropout rate
LEARNING_RATE = 0.001  # Learning rate for Adam optimizer

# Strategy parameters
BOLLINGER_WINDOW = 20   # Window for Bollinger Bands
VOLUME_THRESHOLD = 1.2  # Volume confirmation threshold
STD_DEV = 2.0          # Standard deviations for Bollinger Bands

# Trading parameters
MIN_RETURN = 0.005      # Minimum return to trigger a trade
MAX_HOLD_DAYS = 10     # Maximum holding period
INITIAL_CAPITAL = 100000  # Initial investment amount

# Data parameters
TRAIN_TEST_SPLIT_DATE = '2021-01-01'  # Date to split train/test data