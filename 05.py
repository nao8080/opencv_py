# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # データセットを読み込む
    faces   = fetch_olivetti_faces()
    images  = faces.images.reshape((len(faces.images), -1))
    targets = faces.target
    image_size = faces.images.shape[1:]

    # データセットを学習用とテスト用に分割する
    test_size = 5
    train_data, test_data = train_test_split(images, test_size=test_size, random_state=0)

    # 画像の上半分を入力データ、下半分を正解データとする
    n_pixels = images.shape[1]
    train_x = train_data[:, :(n_pixels + 1) // 2]
    train_y = train_data[:, n_pixels // 2:]
    test_x = test_data[:, :(n_pixels + 1) // 2]
    test_y = test_data[:, n_pixels // 2:]

    # 使用する回帰モデルを定義する
    ESTIMATORS = {
        'Linear Regression' : LinearRegression(),
        'Extra trees' : ExtraTreesRegressor(),
        'K-nn' : KNeighborsRegressor(),
    }

    # 回帰モデルの学習＆テスト（予測結果として画像の下半分を出力する）
    pred = { 'true face' : test_y }
    for name, estimator in ESTIMATORS.items():
        estimator.fit(train_x, train_y)
        pred[name] = estimator.predict(test_x)

    # テストで出力した画像をプロットする
    fig, axes = plt.subplots(test_size, len(pred), figsize=(10,10))
    for i in range(test_size):
        for j, (name, data) in enumerate(pred.items()):
            arr = np.hstack((test_x[i], data[i]))
            img = arr.reshape(image_size)
            if i==0:
                axes[i,j].set_title(name)
            axes[i,j].imshow(img, cmap='gray')
            axes[i,j].set_xticks(())
            axes[i,j].set_yticks(())
    plt.subplots_adjust(hspace=0.1, wspace=0.0)
    plt.show()
