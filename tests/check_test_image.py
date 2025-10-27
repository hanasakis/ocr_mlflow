import cv2
import matplotlib.pyplot as plt
import numpy as np


def analyze_test_image(image_path):
    """分析测试图片的质量"""
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("无法读取图片")
        return

    print(f"图片尺寸: {image.shape}")
    print(f"像素值范围: {image.min()} - {image.max()}")

    # 显示原图
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("原图")
    plt.axis("off")

    # 二值化处理
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    plt.subplot(1, 4, 2)
    plt.imshow(binary, cmap="gray")
    plt.title("二值化后")
    plt.axis("off")

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title("检测到的轮廓")
    plt.axis("off")

    # 显示每个数字区域
    processed_image = image.copy()
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 20:  # 过滤小轮廓
            digit_roi = binary[y : y + h, x : x + w]
            digit_resized = cv2.resize(digit_roi, (28, 28))

            plt.subplot(1, 4, 4)
            plt.imshow(digit_resized, cmap="gray")
            plt.title(f"数字 {i+1} 处理结果")
            plt.axis("off")
            break

    plt.tight_layout()
    plt.show()

    print(f"检测到的轮廓数量: {len(contours)}")
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 20:
            print(f"轮廓 {i}: 位置({x},{y}), 尺寸({w}x{h})")


if __name__ == "__main__":
    analyze_test_image("tests/multi_digits.png")
