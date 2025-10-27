import os
import sys

sys.path.append('src')

import argparse

from prediction import DigitPredictor


def main():
    parser = argparse.ArgumentParser(description='手写数字识别')
    parser.add_argument('image_path', help='要识别的图片路径')
    parser.add_argument('--model', choices=['cnn', 'svm'], default='cnn',
                       help='选择使用的模型 (default: cnn)')
    parser.add_argument('--multi', action='store_true',
                       help='识别图片中的多个数字')

    args = parser.parse_args()

    # 检查模型文件是否存在
    if args.model == 'cnn':
        model_path = 'models/cnn_model.h5'
        if not os.path.exists(model_path):
            print("错误: CNN模型文件不存在，请先运行 train.py 训练模型")
            return
    else:
        model_path = 'models/svm_model.pkl'
        if not os.path.exists(model_path):
            print("错误: SVM模型文件不存在，请先运行 train.py 训练模型")
            return

    print(f"=== 使用{args.model.upper()}模型进行数字识别 ===")

    # 加载预测器
    if args.model == 'cnn':
        predictor = DigitPredictor(cnn_model_path=model_path)
    else:
        predictor = DigitPredictor(svm_model_path=model_path)

    if args.multi:
        # 识别多个数字
        print("多数字识别模式...")
        results, processed_image = predictor.predict_multiple_digits(args.image_path, args.model)

        print("识别结果:")
        for i, result in enumerate(results):
            print(f"数字 {i+1}: {result['digit']}, 置信度: {result['confidence']:.4f}")

        # 可视化结果
        predictor.visualize_prediction(args.image_path, results, processed_image)
    else:
        # 识别单个数字
        digit, confidence = predictor.predict_single_image(args.image_path, args.model)
        print(f"预测数字: {digit}")
        print(f"置信度: {confidence:.4f}")

if __name__ == "__main__":
    main()
