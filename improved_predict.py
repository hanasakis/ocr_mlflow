import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class ImprovedDigitPredictor:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print("✓ 模型加载成功")
        else:
            print("❌ 模型文件不存在")

    def advanced_preprocess(self, image_path, target_size=(28, 28)):
        """改进的图像预处理"""
        # 读取图片
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 自动检测背景颜色并相应处理
        height, width = image.shape
        corner_pixels = [
            image[0, 0], image[0, width-1],
            image[height-1, 0], image[height-1, width-1]
        ]
        avg_corner = np.mean(corner_pixels)

        # 如果角落较亮，说明是白底黑字，需要反转
        if avg_corner > 127:
            image = 255 - image
            print("检测到白底黑字图片，已自动反转")

        # 多种二值化方法尝试
        results = []

        # 方法1: 简单阈值
        _, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        results.append(('简单阈值', thresh1))

        # 方法2: Otsu阈值
        _, thresh2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(('Otsu阈值', thresh2))

        # 方法3: 自适应阈值
        thresh3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        results.append(('自适应阈值', thresh3))

        return image, results

    def extract_digits(self, binary_image, min_area=50):
        """提取数字区域"""
        # 查找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        positions = []

        for contour in contours:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤太小的区域
            if w * h > min_area and w > 5 and h > 10:
                # 提取数字区域
                digit_roi = binary_image[y:y+h, x:x+w]

                # 调整大小，保持宽高比
                if w > h:
                    new_w = 20
                    new_h = int(20 * h / w)
                else:
                    new_h = 20
                    new_w = int(20 * w / h)

                digit_resized = cv2.resize(digit_roi, (new_w, new_h))

                # 创建28x28的画布，将数字放在中心
                canvas = np.zeros((28, 28), dtype=np.uint8)
                x_offset = (28 - new_w) // 2
                y_offset = (28 - new_h) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

                digits.append(canvas)
                positions.append((x, y, w, h))

        return digits, positions

    def predict_multiple(self, image_path):
        """改进的多数字预测"""
        if self.model is None:
            return [], None

        try:
            # 高级预处理
            original, processed_versions = self.advanced_preprocess(image_path)

            best_results = []
            best_confidence = 0

            # 尝试不同的预处理方法
            for method_name, processed_img in processed_versions:
                print(f"\n尝试方法: {method_name}")

                # 提取数字
                digits, positions = self.extract_digits(processed_img)

                if len(digits) == 0:
                    print("未检测到数字")
                    continue

                # 准备输入数据
                digits_array = np.array(digits).astype('float32') / 255.0
                digits_array = digits_array.reshape(-1, 28, 28, 1)

                # 预测
                predictions = self.model.predict(digits_array, verbose=0)

                # 计算平均置信度
                avg_confidence = np.mean(np.max(predictions, axis=1))

                print(f"检测到 {len(digits)} 个数字，平均置信度: {avg_confidence:.4f}")

                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_results = []

                    for i, (pred, pos) in enumerate(zip(predictions, positions)):
                        digit = np.argmax(pred)
                        confidence = np.max(pred)
                        best_results.append({
                            'digit': digit,
                            'confidence': confidence,
                            'position': pos,
                            'method': method_name
                        })

            # 按x坐标排序
            best_results.sort(key=lambda x: x['position'][0])

            return best_results, original

        except Exception as e:
            print(f"预测错误: {e}")
            return [], None

    def visualize_results(self, image_path, results, original_image):
        """可视化结果"""
        # 创建彩色图像用于标注
        if len(original_image.shape) == 2:
            output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            output_image = original_image.copy()

        for i, result in enumerate(results):
            x, y, w, h = result['position']
            digit = result['digit']
            confidence = result['confidence']

            # 绘制边界框
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)  # 绿色或橙色
            cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)

            # 添加标签
            label = f"{digit}({confidence:.2f})"
            cv2.putText(output_image, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示结果
        cv2.imshow('识别结果', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # 检查模型
    model_path = 'models/cnn_model.h5'
    if not os.path.exists(model_path):
        print("模型文件不存在，请先训练模型")
        return

    # 创建预测器
    predictor = ImprovedDigitPredictor(model_path)

    # 测试不同的图片
    test_images = [
        'tests/mnist_style.png',
        'tests/white_bg_style.png',
        'tests/noisy_style.png',
        'tests/multi_digits.png'
    ]

    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n{'='*50}")
            print(f"测试图片: {test_image}")
            print(f"{'='*50}")

            results, original = predictor.predict_multiple(test_image)

            if results:
                print("\n识别结果:")
                for i, result in enumerate(results):
                    print(f"数字 {i+1}: {result['digit']}, 置信度: {result['confidence']:.4f}")

                # 可视化结果
                predictor.visualize_results(test_image, results, original)
            else:
                print("未识别到数字")
        else:
            print(f"测试图片不存在: {test_image}")

if __name__ == "__main__":
    main()
