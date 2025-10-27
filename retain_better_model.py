# retain_better_model.py（新增MLflow代码）
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import git
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

from config import config  # 导入配置

def get_git_commit_hash():
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]
    except Exception:
        return "unknown-commit"

def get_dvc_data_version(data_dir_dvc_path):
    try:
        import dvc.api
        return dvc.api.repo_version(data_dir_dvc_path)[:8]
    except Exception:
        return "unknown-dvc-version"

def create_improved_model():
    """创建改进的CNN模型（原有）"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def augment_data(images, labels):
    """数据增强（原有）"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    return datagen.flow(images.reshape(images.shape[0], 28, 28, 1), labels, batch_size=32)

def main():
    print("=== 训练改进的CNN模型 ===")
    
    # 新增：初始化MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    # 新增：获取代码和数据版本
    git_commit = get_git_commit_hash()
    raw_data_version = get_dvc_data_version("data/raw.dvc")
    
    # 创建目录（原有，改用config路径）
    os.makedirs(str(config.MODELS_DIR), exist_ok=True)
    
    # 加载数据（原有，改用config路径）
    preprocessor = DataPreprocessor(  # 新增：用DataPreprocessor加载数据
        raw_data_path=str(config.RAW_DATA_DIR),
        processed_data_path=str(config.PROCESSED_DATA_DIR)
    )
    (x_train, y_train), (x_test, y_test) = preprocessor.load_mnist_data()
    
    # 数据预处理（原有）
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(f"训练集: {x_train.shape}")
    print(f"测试集: {x_test.shape}")
    
    # 创建模型（原有）
    model = create_improved_model()
    model.summary()
    
    # 回调函数（原有）
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    # 新增：MLflow记录改进模型
    with mlflow.start_run(run_name=f"Improved-CNN-{git_commit}") as run:
        # 记录元数据和超参数
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("raw_data_version", raw_data_version)
        mlflow.log_param("model_type", "Improved-CNN")
        mlflow.log_param("batch_size", config.BATCH_SIZE)
        mlflow.log_param("epochs", 20)  # 硬编码，可改为从config获取
        mlflow.log_param("data_augmentation", True)
        
        # 训练模型（原有）
        print("开始训练...")
        history = model.fit(
            x_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=20,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型（原有，记录metrics）
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\n测试准确率: {test_acc:.4f}")
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
        
        # 记录模型和训练历史图
        signature = infer_signature(x_test[:10], model.predict(x_test[:10]))
        mlflow.keras.log_model(
            model, 
            artifact_path="improved-cnn-model",
            signature=signature,
            registered_model_name="OCR-Improved-CNN-Model"
        )
        
        # 保存训练历史图到MLflow
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='训练准确率')
        plt.plot(history.history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.legend()
        plot_path = str(config.BASE_DIR / "improved_training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path, "training-plots")  # 上传到MLflow
        os.remove(plot_path)  # 本地删除，避免提交git
    
    # 保存模型到本地（原有，改用config路径）
    model_path = str(config.MODELS_DIR / "improved_cnn_model.h5")
    model.save(model_path)
    print(f"✓ 改进的模型已保存: {model_path}")
    
    print("\n=== 训练完成 ===")
    print(f"MLflow运行链接: {config.MLFLOW_TRACKING_URI}/#/experiments/{mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME).experiment_id}")

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor  # 局部导入，避免循环依赖
    main()