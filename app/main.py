#!/usr/bin/env python3
"""
OCR数字识别主应用
整合版本 - 支持CNN和SVM模型，单数字和多数字识别
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

# 添加src目录到Python路径
src_dir = root_dir / 'src'
sys.path.insert(0, str(src_dir))

# 导入配置和预测模块
try:
    from config import config
    from prediction import DigitPredictor
    print("✓ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("尝试备用导入方案...")
    
    # 备用导入方法
    import importlib.util
    
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

    def __init__(self, model_type='cnn'):
        self.predictor = None
        self.model_type = model_type
        self.setup()

    def setup(self):
        """应用初始化"""
        logger.info(f"初始化OCR应用 - 使用{self.model_type.upper()}模型")

        # 根据模型类型检查模型文件
        if self.model_type == 'cnn':
            model_path = config.MODEL_PATH
        else:  # svm
            model_path = config.SVM_MODEL_PATH

        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            print(f"❌ {self.model_type.upper()}模型文件不存在，请先运行训练脚本")
            print("运行: python train.py")
            return False

        # 加载预测器
        try:
            if self.model_type == 'cnn':
                self.predictor = DigitPredictor(cnn_model_path=str(model_path))
            else:
                self.predictor = DigitPredictor(svm_model_path=str(model_path))
                
            logger.info(f"{self.model_type.upper()}模型加载成功")
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
            digit, confidence = self.predictor.predict_single_image(image_path, self.model_type)
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
            results, processed_image = self.predictor.predict_multiple_digits(image_path, self.model_type)
            return results, processed_image
        except Exception as e:
            logger.error(f"预测失败: {e}")
            print(f"❌ 预测失败: {e}")
            return [], None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OCR数字识别应用')
    parser.add_argument('image_path', help='要识别的图片路径')
    parser.add_argument('--model', choices=['cnn', 'svm'], default='cnn',
                       help='选择使用的模型 (default: cnn)')
    parser.add_argument('--multi', action='store_true',
                       help='识别图片中的多个数字')

    args = parser.parse_args()

    # 验证图片路径
    if not os.path.exists(args.image_path):
        print(f"❌ 图片不存在: {args.image_path}")
        return

    try:
        # 创建应用实例
        app = OCRApplication(model_type=args.model)

        # 检查是否初始化成功
        if app.predictor is None:
            return

        print(f"=== 使用{args.model.upper()}模型进行数字识别 ===")

        if args.multi:
            # 多数字识别模式
            print("多数字识别模式...")
            results, processed_image = app.predict_multiple(args.image_path)
            
            if results:
                print("🎯 识别结果:")
                for i, result in enumerate(results):
                    print(f"数字 {i+1}: {result['digit']}, 置信度: {result['confidence']:.4f}")
                
                # 尝试可视化结果（在容器中可能无法显示，但可以保存）
                try:
                    if hasattr(app.predictor, 'visualize_prediction'):
                        app.predictor.visualize_prediction(args.image_path, results, processed_image)
                        print("✓ 结果可视化已完成")
                except Exception as e:
                    print(f"⚠️ 可视化失败（可能在容器环境中）: {e}")
            else:
                print("❌ 未识别到任何数字")
        else:
            # 单数字识别模式
            digit, confidence = app.predict_single(args.image_path)
            if digit is not None:
                print(f"🎯 识别结果: 数字 {digit}")
                print(f"✅ 置信度: {confidence:.4f}")
            else:
                print("❌ 识别失败")

    except Exception as e:
        logger.error(f"应用执行失败: {e}")
        print(f"❌ 应用执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()