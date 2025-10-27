import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model_training import ModelTrainer


class TestModelTrainer:
    """测试模型训练类"""

    def setup_method(self):
        self.trainer = ModelTrainer()

    def test_create_cnn_model(self):
        """测试CNN模型创建"""
        model = self.trainer.create_cnn_model()

        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape == (None, 28, 28, 1)
        assert model.output_shape == (None, 10)

    @patch("model_training.tf.keras.models.Sequential")
    def test_train_cnn(self, mock_sequential):
        """测试CNN模型训练"""
        # 创建模拟数据
        x_train = np.random.random((100, 28, 28, 1))
        y_train = np.random.randint(0, 10, 100)
        x_test = np.random.random((20, 28, 28, 1))
        y_test = np.random.randint(0, 10, 20)

        # 模拟模型
        mock_model = Mock()
        mock_model.fit.return_value = Mock(history={"accuracy": [0.9], "loss": [0.3]})
        mock_sequential.return_value = mock_model

        trainer = ModelTrainer()
        trainer.cnn_model = mock_model

        history = trainer.train_cnn(x_train, y_train, x_test, y_test, epochs=1)

        assert history is not None
        mock_model.fit.assert_called_once()

    def test_evaluate_model_cnn(self):
        """测试CNN模型评估"""
        # 创建模拟模型和数据
        mock_model = Mock()
        mock_model.evaluate.return_value = [0.5, 0.9]  # [loss, accuracy]
        mock_model.predict.return_value = np.random.random((20, 10))

        x_test = np.random.random((20, 28, 28, 1))
        y_test = np.random.randint(0, 10, 20)

        trainer = ModelTrainer()
        trainer.cnn_model = mock_model

        y_pred, accuracy = trainer.evaluate_model(mock_model, x_test, y_test, "cnn")

        assert accuracy == 0.9
        assert len(y_pred) == 20
