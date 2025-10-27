import os
import sys

import cv2
import numpy as np
import pytest

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """测试数据预处理类"""

    def setup_method(self):
        """测试设置"""
        self.preprocessor = DataPreprocessor()

    def test_preprocess_images_normalization(self):
        """测试图片归一化"""
        # 创建测试数据
        test_images = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)

        # 预处理
        processed = self.preprocessor.preprocess_images(test_images)

        # 验证归一化到0-1
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
        assert processed.dtype == np.float32

    def test_preprocess_images_shape(self):
        """测试图片形状处理"""
        test_images = np.random.randint(0, 256, (5, 28, 28), dtype=np.uint8)

        # 测试CNN输入形状
        processed_cnn = self.preprocessor.preprocess_images(test_images, flatten=False)
        assert processed_cnn.shape == (5, 28, 28, 1)

        # 测试SVM输入形状
        processed_svm = self.preprocessor.preprocess_images(test_images, flatten=True)
        assert processed_svm.shape == (5, 784)

    def test_preprocess_custom_image(self, tmp_path):
        """测试自定义图片预处理"""
        # 创建测试图片
        test_img_path = tmp_path / "test.png"
        img = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.imwrite(str(test_img_path), img)

        # 预处理
        processed = self.preprocessor.preprocess_custom_image(str(test_img_path))

        assert processed.shape == (1, 28, 28, 1)
        assert processed.dtype == np.float32

    def test_preprocess_custom_image_file_not_found(self):
        """测试文件不存在的情况"""
        with pytest.raises(ValueError, match="无法读取图片"):
            self.preprocessor.preprocess_custom_image("nonexistent.png")
