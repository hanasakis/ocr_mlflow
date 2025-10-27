# train.py（新增MLflow相关代码）
import os
import sys
import git
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
sys.path.append('src')  # 确保路径正确

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from config import config  # 导入配置

def get_git_commit_hash():
    """获取当前Git提交哈希（记录代码版本）"""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]  # 取前8位简化显示
    except Exception:
        return "unknown-commit"

def get_dvc_data_version(data_dir_dvc_path):
    """获取DVC数据版本（记录数据集版本）"""
    try:
        import dvc.api
        # 通过DVC API获取数据版本（hash）
        with dvc.api.open(data_dir_dvc_path) as f:
            return dvc.api.repo_version(data_dir_dvc_path)[:8]
    except Exception:
        return "unknown-dvc-version"

def main():
    # 新增：初始化MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    # 新增：获取代码版本和数据集版本
    git_commit = get_git_commit_hash()
    raw_data_version = get_dvc_data_version("data/raw.dvc")
    processed_data_version = get_dvc_data_version("data/processed.dvc")
    
    # 创建必要的目录（原有）
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    print("=== 手写数字OCR识别模型训练 ===")

    # 数据预处理（原有）
    print("1. 加载和预处理数据...")
    preprocessor = DataPreprocessor(
        raw_data_path=str(config.RAW_DATA_DIR),  # 改用config路径
        processed_data_path=str(config.PROCESSED_DATA_DIR)  # 改用config路径
    )
    (x_train, y_train), (x_test, y_test) = preprocessor.load_mnist_data()
    x_train_cnn = preprocessor.preprocess_images(x_train)
    x_test_cnn = preprocessor.preprocess_images(x_test)
    x_train_svm, x_test_svm = preprocessor.preprocess_for_svm(x_train, x_test)
    print(f"训练集形状: {x_train_cnn.shape}")
    print(f"测试集形状: {x_test_cnn.shape}")

    # 模型训练（原有，新增MLflow上下文）
    print("\n2. 训练CNN模型...")
    trainer = ModelTrainer()
    
    # 新增：MLflow开始运行（记录CNN模型）
    with mlflow.start_run(run_name=f"CNN-Train-{git_commit}") as cnn_run:
        # 记录元数据（代码版本、数据集版本）
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("raw_data_version", raw_data_version)
        mlflow.log_param("processed_data_version", processed_data_version)
        
        # 记录超参数（从config获取）
        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("batch_size", config.BATCH_SIZE)
        mlflow.log_param("epochs", config.EPOCHS)
        mlflow.log_param("learning_rate", config.LEARNING_RATE)
        
        # 训练CNN模型（原有）
        cnn_history = trainer.train_cnn(
            x_train_cnn, y_train, x_test_cnn, y_test,
            epochs=config.EPOCHS, batch_size=config.BATCH_SIZE
        )
        trainer.plot_training_history(cnn_history)
        
        # 评估CNN模型（原有，记录metrics）
        print("\n3. 评估CNN模型...")
        cnn_pred, cnn_acc = trainer.evaluate_model(trainer.cnn_model, x_test_cnn, y_test, 'cnn')
        mlflow.log_metric("test_accuracy", cnn_acc)
        mlflow.log_metric("train_accuracy", cnn_history.history["accuracy"][-1])
        mlflow.log_metric("val_accuracy", cnn_history.history["val_accuracy"][-1])
        
        # 记录模型（保存到MLflow，含签名）
        signature = infer_signature(x_test_cnn[:10], trainer.cnn_model.predict(x_test_cnn[:10]))
        mlflow.keras.log_model(
            trainer.cnn_model, 
            artifact_path="cnn-model", 
            signature=signature,
            registered_model_name="OCR-CNN-Model"  # 注册模型（版本控制）
        )

    # 训练SVM模型（原有，新增MLflow记录）
    print("\n4. 训练SVM模型...")
    with mlflow.start_run(run_name=f"SVM-Train-{git_commit}") as svm_run:
        # 记录元数据和超参数
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("raw_data_version", raw_data_version)
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", "rbf")
        
        # 训练SVM模型（原有）
        trainer.train_svm(x_train_svm, y_train)
        
        # 评估SVM模型（原有，记录metrics）
        print("\n5. 评估SVM模型...")
        svm_pred, svm_acc = trainer.evaluate_model(trainer.svm_model, x_test_svm, y_test, 'svm')
        mlflow.log_metric("test_accuracy", svm_acc)
        
        # 记录SVM模型
        mlflow.sklearn.log_model(
            trainer.svm_model, 
            artifact_path="svm-model",
            registered_model_name="OCR-SVM-Model"
        )

    # 保存模型到本地（原有）
    print("\n6. 保存模型...")
    trainer.save_models(
        cnn_path=str(config.MODEL_PATH),  # 改用config路径
        svm_path=str(config.MODELS_DIR / "svm_model.pkl")
    )

    print(f"\n=== 训练完成 ===")
    print(f"CNN模型准确率: {cnn_acc:.4f}")
    print(f"SVM模型准确率: {svm_acc:.4f}")
    print(f"MLflow运行链接: {config.MLFLOW_TRACKING_URI}/#/experiments/{mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME).experiment_id}")

if __name__ == "__main__":
    main()