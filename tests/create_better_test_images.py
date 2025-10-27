import os

import cv2
import numpy as np


def create_mnist_style_digit(digit, size=28):
    """创建MNIST风格的单个数字图片"""
    # 创建黑色背景（与MNIST一致）
    image = np.zeros((size, size), dtype=np.uint8)

    # 在中心位置绘制数字
    text_size = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    x = (size - text_size[0]) // 2
    y = (size + text_size[1]) // 2

    cv2.putText(image, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

    return image


def create_better_test_image(digits, filename="better_multi_digits.png"):
    """创建更好的测试图片"""
    # 图片参数
    digit_size = 28
    spacing = 10
    margin = 20

    # 计算画布尺寸
    canvas_width = len(digits) * digit_size + (len(digits) - 1) * spacing + 2 * margin
    canvas_height = digit_size + 2 * margin

    # 创建画布（黑色背景，与MNIST一致）
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # 放置每个数字
    for i, digit in enumerate(digits):
        # 创建单个数字
        digit_img = create_mnist_style_digit(digit)

        # 计算位置
        x_start = margin + i * (digit_size + spacing)
        y_start = margin

        # 将数字粘贴到画布上
        canvas[y_start : y_start + digit_size, x_start : x_start + digit_size] = (
            digit_img
        )

    # 保存图片
    os.makedirs("tests", exist_ok=True)
    cv2.imwrite(f"tests/{filename}", canvas)
    print(f"✓ 更好的测试图片已创建: tests/{filename}")

    return f"tests/{filename}"


def create_varied_test_images():
    """创建多种风格的测试图片"""
    digits = [1, 2, 3, 4, 5]

    # 1. MNIST风格（黑底白字）
    create_better_test_image(digits, "mnist_style.png")

    # 2. 白底黑字风格（需要反转）
    img_path = create_better_test_image(digits, "white_bg_style.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_inverted = 255 - img  # 反转为白底黑字
    cv2.imwrite("tests/white_bg_style.png", img_inverted)

    # 3. 添加噪声的版本
    img = cv2.imread("tests/mnist_style.png", cv2.IMREAD_GRAYSCALE)
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img_noisy = cv2.add(img, noise)
    cv2.imwrite("tests/noisy_style.png", img_noisy)

    print("✓ 多种风格的测试图片已创建完成")


if __name__ == "__main__":
    create_varied_test_images()
