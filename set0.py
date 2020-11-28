import numpy as np
import cv2
import matplotlib.pyplot as plt

# 画像読み込み
img_bgr = cv2.imread('Lena.jpg') 
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(img_bgr)
plt.subplot(122)
plt.imshow(img_rgb)


sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img_bgr)
img_kp = np.zeros_like(img_bgr)
img_kp = cv2.drawKeypoints(img_rgb,kp,img_kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_kp)







