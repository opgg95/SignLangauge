# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 18:31:22 2023

@author: KIBWA_23
"""

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


max_num_hands = 1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

f = open('text.txt', 'w')

file = np.genfromtxt('dataSetEnglish.txt', delimiter=',')
data = file[:, :-1]
target = file[:, -1]
angle = data.astype(np.float32)
label = target.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(angle, label, test_size=0.3, random_state=45)

knn = cv2.ml.KNearest_create()
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

ret, results,neighbouts,dist=knn.findNearest(x_test,3)

index=int(results[0][0])
print(index)

print("Train dataset size:", len(y_train))
print("Test dataset size:", len(y_test))

# 정확도 계산
correct_predictions = np.sum(results.squeeze() == y_test)
total_predictions = y_test.shape[0]
accuracy = correct_predictions / total_predictions * 100

# 정확도 출력
print("정확도: {:.2f}%".format(accuracy))

# 혼동 행렬 계산
cm = confusion_matrix(y_test, results.squeeze())

# 혼동 행렬 출력
print("혼동 행렬:")
print(cm)