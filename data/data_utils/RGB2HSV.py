import cv2
import numpy as np

BGR = [[180, 157, 146]]
# BGR = [[134, 130, 129]]
# BGR = [[161, 146, 140]]
# BGR = [[216, 166, 143]]
# BGR = [[138, 132, 131]]
# 假设你有一个 RGB 图像
rgb_image = np.array([BGR], dtype=np.uint8)  # 示例 RGB 图像

# 将 RGB 图像转换为 HSV 色彩空间
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

print("BGR ", )
print("HSV ", hsv_image)
print()
