from pathlib import Path

# 项目路径
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DATA_DIR = ROOT_DIR / "CIFAR10"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs_CIFAR10"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

# 数据与训练超参数
BATCH_SIZE = 64
NUM_WORKERS = 0  # Windows 下从 0 开始，避免多进程 DataLoader 配置问题
TRAIN_SIZE = 45_000
VAL_SIZE = 5_000
RANDOM_SEED = 42

EPOCHS = 10
LEARNING_RATE = 1e-2
LOG_INTERVAL = 100  # log间隔
