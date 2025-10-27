#!/usr/bin/env python3
"""
OCR数字识别主应用
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

# 添加src目录到Python路径
src_dir = root_dir / 'src'
sys.path.insert(0, str(src_dir))

try:
    from config import config
    from prediction import DigitPredictor
    print("✓ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("尝试备用导入方案...")

    # 备用导入方法
    import importlib.util
    import os

    # 导入config
    config_spec = importlib.util.spec_from_file_location("config", root_dir / "config.py")
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    config = config_module.config

    # 导入DigitPredictor
    prediction_spec = importlib.util.spec_from_file_location("prediction", src_dir / "prediction.py")
    prediction_module = importlib.util.module_from_spec(prediction_spec)
    prediction_spec.loader.exec_module(prediction_module)
    DigitPredictor = prediction_module.DigitPredictor

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRApplication:
    """OCR应用主类"""

    def __init__(self):
        self.predictor = None
        self.setup()

    def setup(self):
        """应用初始化"""
        logger.info("初始化OCR应用")

        # 检查模型文件
        if not config.MODEL_PATH.exists():
            logger.error(f"模型文件不存在: {config.MODEL_PATH}")
            print("❌ 模型文件不存在，请先运行训练脚本")
            print("运行: python train.py")
            return False

        # 加载预测器
        try:
            self.predictor = DigitPredictor(cnn_model_path=str(config.MODEL_PATH))
            logger.info("模型加载成功")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            print(f"❌ 模型加载失败: {e}")
            return False

    def predict_single(self, image_path):
        """预测单张图片"""
        if self.predictor is None:
            print("❌ 预测器未初始化")
            return None, None

        logger.info(f"预测图片: {image_path}")

        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            print(f"❌ 图片不存在: {image_path}")
            return None, None

        try:
            digit, confidence = self.predictor.predict_single_image(image_path)
            return digit, confidence
        except Exception as e:
            logger.error(f"预测失败: {e}")
            print(f"❌ 预测失败: {e}")
            return None, None

    def predict_multiple(self, image_path):
        """预测多数字图片"""
        if self.predictor is None:
            print("❌ 预测器未初始化")
            return [], None

        logger.info(f"预测多数字图片: {image_path}")

        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            print(f"❌ 图片不存在: {image_path}")
            return [], None

        try:
            results, processed_image = self.predictor.predict_multiple_digits(image_path)
            return results, processed_image
        except Exception as e:
            logger.error(f"预测失败: {e}")
            print(f"❌ 预测失败: {e}")
            return [], None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OCR数字识别应用')
    parser.add_argument('image_path', help='要识别的图片路径')
    parser.add_argument('--mode', choices=['single', 'multi'], default='single',
                       help='识别模式: single(单数字)或multi(多数字)')
    parser.add_argument('--model', choices=['cnn', 'svm'], default='cnn',
                       help='使用的模型')

    args = parser.parse_args()

    try:
        app = OCRApplication()

        # 检查是否初始化成功
        if app.predictor is None:
            return

        if args.mode == 'single':
            digit, confidence = app.predict_single(args.image_path)
            if digit is not None:
                print(f"🎯 识别结果: 数字 {digit}")
                print(f"✅ 置信度: {confidence:.4f}")
            else:
                print("❌ 识别失败")
        else:
            results, _ = app.predict_multiple(args.image_path)
            if results:
                print("识别结果:")
                for i, result in enumerate(results):
                    print(f"数字 {i+1}: {result['digit']}, 置信度: {result['confidence']:.4f}")
            else:
                print("❌ 未识别到任何数字")

    except Exception as e:
        logger.error(f"应用执行失败: {e}")
        print(f"❌ 应用执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
