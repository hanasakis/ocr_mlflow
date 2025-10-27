import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from tensorflow.keras import layers, models


class ModelTrainer:
    def __init__(self):
        self.cnn_model = None
        self.svm_model = None

    def create_cnn_model(self, input_shape=(28, 28, 1), num_classes=10):
        """创建CNN模型"""
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train_cnn(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
        """训练CNN模型"""
        self.cnn_model = self.create_cnn_model()

        # 添加早停回调
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]

        history = self.cnn_model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def train_svm(self, x_train, y_train):
        """训练SVM模型"""
        self.svm_model = SVC(kernel="rbf", gamma="scale", C=1.0, probability=True)
        self.svm_model.fit(x_train, y_train)
        return self.svm_model

    def evaluate_model(self, model, x_test, y_test, model_type="cnn"):
        """评估模型性能"""
        if model_type == "cnn":
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            y_pred = np.argmax(model.predict(x_test), axis=1)
        else:
            y_pred = model.predict(x_test)
            test_acc = model.score(x_test, y_test)

        print(f"测试准确率: {test_acc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        return y_pred, test_acc

    def plot_training_history(self, history):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 准确率曲线
        ax1.plot(history.history["accuracy"], label="训练准确率")
        ax1.plot(history.history["val_accuracy"], label="验证准确率")
        ax1.set_title("模型准确率")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # 损失曲线
        ax2.plot(history.history["loss"], label="训练损失")
        ax2.plot(history.history["val_loss"], label="验证损失")
        ax2.set_title("模型损失")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def save_models(
        self, cnn_path="../models/cnn_model.h5", svm_path="../models/svm_model.pkl"
    ):
        """保存模型"""
        if self.cnn_model:
            self.cnn_model.save(cnn_path)
            print(f"CNN模型已保存到: {cnn_path}")

        if self.svm_model:
            joblib.dump(self.svm_model, svm_path)
            print(f"SVM模型已保存到: {svm_path}")


if __name__ == "__main__":
    # 这里主要用于测试，实际训练通过train.py进行
    print("模型训练模块加载成功")
