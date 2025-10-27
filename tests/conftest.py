import os
import tempfile

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image_data():
    """提供样本图片数据"""
    return np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)


@pytest.fixture
def sample_labels():
    """提供样本标签"""
    return np.random.randint(0, 10, 10)


@pytest.fixture
def temp_image_file():
    """创建临时图片文件"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # 创建测试图片
        img = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.putText(img, "7", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
        cv2.imwrite(f.name, img)
        yield f.name
    # 清理
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def mock_model():
    """创建模拟模型"""

    class MockModel:
        def predict(self, x, verbose=0):
            return np.random.random((x.shape[0], 10))

        def evaluate(self, x, y, verbose=0):
            return [0.5, 0.9]  # loss, accuracy

    return MockModel()
