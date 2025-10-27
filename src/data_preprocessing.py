import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, raw_data_path="data/raw", processed_data_path="data/processed"):
        self.scaler = None
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

        # 创建处理后的数据目录
        os.makedirs(processed_data_path, exist_ok=True)

    def load_mnist_data(self):
        """从本地路径加载MNIST数据集"""
        try:
            # 检查是否有合并的npz文件
            npz_path = os.path.join(self.raw_data_path, "mnist.npz")
            if os.path.exists(npz_path):
                print("从npz文件加载MNIST数据...")
                with np.load(npz_path, allow_pickle=True) as f:
                    x_train, y_train = f["x_train"], f["y_train"]
                    x_test, y_test = f["x_test"], f["y_test"]
            else:
                # 检查是否有四个单独的文件
                train_images_path = os.path.join(
                    self.raw_data_path, "train-images-idx3-ubyte.gz"
                )
                train_labels_path = os.path.join(
                    self.raw_data_path, "train-labels-idx1-ubyte.gz"
                )
                test_images_path = os.path.join(
                    self.raw_data_path, "t10k-images-idx3-ubyte.gz"
                )
                test_labels_path = os.path.join(
                    self.raw_data_path, "t10k-labels-idx1-ubyte.gz"
                )

                if all(
                    os.path.exists(p)
                    for p in [
                        train_images_path,
                        train_labels_path,
                        test_images_path,
                        test_labels_path,
                    ]
                ):
                    print("从原始IDX格式文件加载MNIST数据...")
                    x_train = self._load_idx_file(train_images_path)
                    y_train = self._load_idx_file(train_labels_path)
                    x_test = self._load_idx_file(test_images_path)
                    y_test = self._load_idx_file(test_labels_path)
                else:
                    # 如果本地文件不存在，尝试从网络下载
                    print("本地文件不存在，尝试从网络下载...")
                    from tensorflow.keras.datasets import mnist

                    (x_train, y_train), (x_test, y_test) = mnist.load_data()

                    # 保存到本地供下次使用
                    self._save_mnist_locally(x_train, y_train, x_test, y_test)

            print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")
            return (x_train, y_train), (x_test, y_test)

        except Exception as e:
            print(f"加载MNIST数据失败: {e}")
            # 尝试使用tensorflow的备用方法
            try:
                from tensorflow.keras.datasets import mnist

                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                return (x_train, y_train), (x_test, y_test)
            except:
                raise Exception("无法加载MNIST数据集，请检查数据文件路径")

    def _load_idx_file(self, file_path):
        """加载IDX格式文件"""
        with open(file_path, "rb") as f:
            # 读取魔数
            magic_number = int.from_bytes(f.read(4), byteorder="big")
            # 读取维度数量
            num_dims = magic_number & 0xFF

            # 读取维度大小
            dims = []
            for i in range(num_dims):
                dims.append(int.from_bytes(f.read(4), byteorder="big"))

            # 读取数据
            data = np.frombuffer(f.read(), dtype=np.uint8)

            # 重塑数据形状
            if num_dims == 1:
                return data
            else:
                return data.reshape(*dims)

    def _save_mnist_locally(self, x_train, y_train, x_test, y_test):
        """保存MNIST数据到本地"""
        npz_path = os.path.join(self.raw_data_path, "mnist.npz")
        os.makedirs(self.raw_data_path, exist_ok=True)
        np.savez(
            npz_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )
        print(f"MNIST数据已保存到: {npz_path}")

    def preprocess_images(self, images, labels=None, flatten=False):
        """
        预处理图片数据
        """
        # 归一化到0-1范围
        images_normalized = images.astype("float32") / 255.0

        if flatten:
            # 为传统机器学习方法展平图片
            n_samples = images_normalized.shape[0]
            images_flattened = images_normalized.reshape((n_samples, -1))
            return images_flattened
        else:
            # 为CNN添加通道维度
            if len(images_normalized.shape) == 3:  # 如果是灰度图，添加通道维度
                images_reshaped = images_normalized.reshape(*images_normalized.shape, 1)
            else:
                images_reshaped = images_normalized
            return images_reshaped

    def preprocess_for_svm(self, x_train, x_test):
        """为SVM准备数据"""
        # 展平图片
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)

        # 标准化
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train_flat)
        x_test_scaled = self.scaler.transform(x_test_flat)

        return x_train_scaled, x_test_scaled

    def preprocess_custom_image(self, image_path, target_size=(28, 28)):
        """预处理自定义图片"""
        # 读取图片
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 调整大小
        image_resized = cv2.resize(image, target_size)

        # 二值化（确保背景为黑色，数字为白色）
        _, image_binary = cv2.threshold(image_resized, 127, 255, cv2.THRESH_BINARY_INV)

        # 归一化
        image_normalized = image_binary.astype("float32") / 255.0

        return image_normalized.reshape(1, 28, 28, 1)

    def save_processed_data(self, x_train, y_train, x_test, y_test, suffix=""):
        """保存预处理后的数据"""
        # 构建文件路径
        if suffix:
            suffix = "_" + suffix

        x_train_path = os.path.join(self.processed_data_path, f"x_train{suffix}.npy")
        y_train_path = os.path.join(self.processed_data_path, f"y_train{suffix}.npy")
        x_test_path = os.path.join(self.processed_data_path, f"x_test{suffix}.npy")
        y_test_path = os.path.join(self.processed_data_path, f"y_test{suffix}.npy")
        scaler_path = os.path.join(self.processed_data_path, f"scaler{suffix}.pkl")

        # 保存数据
        np.save(x_train_path, x_train)
        np.save(y_train_path, y_train)
        np.save(x_test_path, x_test)
        np.save(y_test_path, y_test)

        # 保存scaler
        if self.scaler is not None:
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

        print(f"预处理数据已保存到: {self.processed_data_path}")
        print(f"文件前缀: {suffix if suffix else 'default'}")

    def load_processed_data(self, suffix=""):
        """加载预处理后的数据"""
        if suffix:
            suffix = "_" + suffix

        x_train_path = os.path.join(self.processed_data_path, f"x_train{suffix}.npy")
        y_train_path = os.path.join(self.processed_data_path, f"y_train{suffix}.npy")
        x_test_path = os.path.join(self.processed_data_path, f"x_test{suffix}.npy")
        y_test_path = os.path.join(self.processed_data_path, f"y_test{suffix}.npy")
        scaler_path = os.path.join(self.processed_data_path, f"scaler{suffix}.pkl")

        # 加载数据
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

        # 加载scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        return (x_train, y_train), (x_test, y_test)

    def visualize_samples(self, images, labels, num_samples=10, title="样本图片"):
        """可视化样本图片"""
        plt.figure(figsize=(12, 6))
        for i in range(min(num_samples, len(images))):
            plt.subplot(2, 5, i + 1)
            if len(images[i].shape) == 3 and images[i].shape[-1] == 1:
                # 如果是单通道图片，压缩通道维度
                plt.imshow(images[i].squeeze(), cmap="gray")
            else:
                plt.imshow(images[i], cmap="gray")
            # 在visualize_samples调用处改为英文标题

            plt.title(f"Label: {labels[i]}")
            plt.axis("off")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 测试数据预处理
    preprocessor = DataPreprocessor(
        raw_data_path="E:/EFREI/cnn_test/data/raw",
        processed_data_path="E:/EFREI/cnn_test/data/processed",
    )

    # 加载MNIST数据
    (x_train, y_train), (x_test, y_test) = preprocessor.load_mnist_data()

    # 可视化原始样本
    # 改为英文字体
    preprocessor.visualize_samples(x_train, y_train, title="Original Samples")

    print(f"原始训练集形状: {x_train.shape}")
    print(f"原始测试集形状: {x_test.shape}")

    # 预处理为CNN格式
    x_train_cnn = preprocessor.preprocess_images(x_train, flatten=False)
    x_test_cnn = preprocessor.preprocess_images(x_test, flatten=False)

    print(f"CNN格式训练集形状: {x_train_cnn.shape}")
    print(f"CNN格式测试集形状: {x_test_cnn.shape}")

    # 保存CNN格式数据
    preprocessor.save_processed_data(
        x_train_cnn, y_train, x_test_cnn, y_test, suffix="cnn"
    )

    # 预处理为SVM格式
    x_train_svm, x_test_svm = preprocessor.preprocess_for_svm(x_train, x_test)

    print(f"SVM格式训练集形状: {x_train_svm.shape}")
    print(f"SVM格式测试集形状: {x_test_svm.shape}")

    # 保存SVM格式数据
    preprocessor.save_processed_data(
        x_train_svm, y_train, x_test_svm, y_test, suffix="svm"
    )

    # 预处理为扁平化格式（传统ML）
    x_train_flat = preprocessor.preprocess_images(x_train, flatten=True)
    x_test_flat = preprocessor.preprocess_images(x_test, flatten=True)

    print(f"扁平化训练集形状: {x_train_flat.shape}")
    print(f"扁平化测试集形状: {x_test_flat.shape}")

    # 保存扁平化格式数据
    preprocessor.save_processed_data(
        x_train_flat, y_train, x_test_flat, y_test, suffix="flat"
    )

    # 可视化预处理后的样本 - 改为英文字体
    preprocessor.visualize_samples(
        x_train_cnn[:10], y_train[:10], title="CNN Format Samples"
    )
