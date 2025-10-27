import cv2
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 改为：
try:
    from .data_preprocessing import DataPreprocessor
except ImportError:
    from data_preprocessing import DataPreprocessor


class DigitPredictor:
    def __init__(self, cnn_model_path=None, svm_model_path=None):
        self.cnn_model = None
        self.svm_model = None
        self.preprocessor = DataPreprocessor()

        if cnn_model_path:
            self.cnn_model = load_model(cnn_model_path)

        if svm_model_path:
            self.svm_model = joblib.load(svm_model_path)

    def predict_single_image(self, image_path, model_type="cnn"):
        """预测单张图片"""
        # 预处理图片
        processed_image = self.preprocessor.preprocess_custom_image(image_path)

        if model_type == "cnn" and self.cnn_model:
            prediction = self.cnn_model.predict(processed_image, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
        elif model_type == "svm" and self.svm_model:
            # SVM需要展平的输入
            flattened_image = processed_image.reshape(1, -1)
            prediction = self.svm_model.predict_proba(flattened_image)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
        else:
            raise ValueError("模型未加载或模型类型不支持")

        return predicted_digit, confidence

    def predict_multiple_digits(self, image_path, model_type="cnn"):
        """识别图片中的多个数字"""
        # 读取图片
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 二值化
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # 寻找轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for i, contour in enumerate(contours):
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤太小的区域
            if w > 10 and h > 20:
                # 提取数字区域
                digit_roi = thresh[y : y + h, x : x + w]

                # 调整大小
                digit_resized = cv2.resize(digit_roi, (28, 28))

                # 归一化
                digit_normalized = digit_resized.astype("float32") / 255.0

                if model_type == "cnn":
                    digit_input = digit_normalized.reshape(1, 28, 28, 1)
                    prediction = self.cnn_model.predict(digit_input, verbose=0)
                    predicted_digit = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                else:
                    digit_input = digit_normalized.reshape(1, -1)
                    prediction = self.svm_model.predict_proba(digit_input)
                    predicted_digit = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])

                results.append(
                    {
                        "digit": predicted_digit,
                        "confidence": confidence,
                        "position": (x, y, w, h),
                        "contour_index": i,
                    }
                )

        # 按x坐标排序（从左到右）
        results.sort(key=lambda x: x["position"][0])

        return results, thresh

    def visualize_prediction(self, image_path, results, processed_image):
        """可视化预测结果"""
        original_image = cv2.imread(image_path)
        output_image = original_image.copy()

        for result in results:
            x, y, w, h = result["position"]
            digit = result["digit"]
            confidence = result["confidence"]

            # 绘制边界框
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 添加标签
            label = f"{digit} ({confidence:.2f})"
            cv2.putText(
                output_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # 显示结果
        cv2.imshow("Original Image", original_image)
        cv2.imshow("Processed Image", processed_image)
        cv2.imshow("Detection Result", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 测试预测功能
    predictor = DigitPredictor("../models/cnn_model.h5")
    digit, confidence = predictor.predict_single_image("../tests/test_image.png")
    print(f"预测数字: {digit}, 置信度: {confidence:.4f}")
