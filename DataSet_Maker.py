# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:08:58 2023

@author: KIBWA_23
"""

import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

max_num_hands=1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

f=open('text.txt','w')

cap=cv2.VideoCapture(0)

startTime=time.time()
sentence=''
recognizeDelay=1
while True:
    ret,img=cap.read()
    if not ret:
        continue
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j,lm in enumerate(res.landmark):
                joint[j]=[lm.x,lm.y,lm.z]
                
            v1=joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2=joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            
            v=v2-v1
            v=v/np.linalg.norm(v,axis=1)[:,np.newaxis]
            compareV1=v[[0,1,2,4,5,6,7,8,9,10,12,13,14,16,17],:]
            compareV2=v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]
            angle=np.arccos(np.einsum('nt,nt->n',compareV1,compareV2))
            
            angle=np.degrees(angle)
            if keyboard.is_pressed('a'): #a를 누를시 현재 데이터(angle)가 txt파일에 저장됨
                for num in angle:
                    num=round(num,6)
                    f.write(str(num))
                    f.write(',')
                f.write("34.000000")
                f.write('\n')
                print('next')
            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
        
    cv2.imshow('HandTracking',img)
    cv2.waitKey(1)
    if keyboard.is_pressed('b'): #  b를 누를시 프로그램 종료
        break
f.close()