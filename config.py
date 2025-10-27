# config.py（新增MLflow配置）
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """应用配置类"""
    # 原有配置...
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    PORT = int(os.getenv('PORT', 8000))
    
    # 路径配置（确保与目录结构一致，无需修改）
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / os.getenv('DATA_DIR', 'data')
    RAW_DATA_DIR = DATA_DIR / 'raw'       # 新增：原始数据路径
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'  # 新增：预处理数据路径
    MODELS_DIR = BASE_DIR / os.getenv('MODELS_DIR', 'models')
    TEST_IMAGES_DIR = BASE_DIR / os.getenv('TEST_IMAGES_DIR', 'tests')
    
    # 模型配置（原有）
    MODEL_PATH = MODELS_DIR / os.getenv('MODEL_PATH', 'cnn_model.h5')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
    EPOCHS = int(os.getenv('EPOCHS', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    
    # API密钥（原有）
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    CUSTOM_API_SECRET = os.getenv('CUSTOM_API_SECRET', '')
    
    # 新增：MLflow配置（从环境变量或DAGsHub获取）
    MLFLOW_TRACKING_URI = os.getenv(
        'MLFLOW_TRACKING_URI', 
        'https://dagshub.com/hanasakis/number_recognize.mlflow'  # 替换为你的URI
    )
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'OCR-Digit-Recognition')
    
    # 确保目录存在（原有，新增RAW/PROCESSED目录）
    RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)
    PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True)
    TEST_IMAGES_DIR.mkdir(exist_ok=True)

config = Config()