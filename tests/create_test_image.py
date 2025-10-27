import os

import cv2
import numpy as np


def create_test_image(digits, filename="test_multi.png", image_size=(200, 100)):
    """创建测试图片"""
    # 创建画布
    canvas = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255

    # 在每个数字的位置绘制
    digit_spacing = image_size[0] // (len(digits) + 1)

    for i, digit in enumerate(digits):
        # 创建数字图片
        digit_img = np.zeros((28, 28), dtype=np.uint8)
        cv2.putText(
            digit_img, str(digit), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2
        )

        # 计算位置
        x_pos = digit_spacing * (i + 1) - 14
        y_pos = image_size[1] // 2 - 14

        # 将数字粘贴到画布上
        canvas[y_pos : y_pos + 28, x_pos : x_pos + 28] = digit_img

    # 保存图片
    os.makedirs("tests", exist_ok=True)
    cv2.imwrite(f"tests/{filename}", canvas)
    print(f"测试图片已保存: tests/{filename}")


if __name__ == "__main__":
    # 创建包含多个数字的测试图片
    create_test_image([1, 2, 3, 4, 5], "multi_digits.png")

    # 创建单个数字测试图片
    create_test_image([7], "test_single.png", image_size=(50, 50))
