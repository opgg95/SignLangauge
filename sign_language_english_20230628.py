# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:17:07 2023

@author: KIBWA_10
"""
import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

max_num_hands=1

gesture={
    0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h',
    8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14: 'o',
    15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v',
    22:'w', 23:'x', 24:'y', 25:'z', 26:'space', 27:'clear', 28:'backspace'
    }
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

file=np.genfromtxt('dataSetEnglish.txt',delimiter=',')
angleFile=file[:,:-1]
labelFile=file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)
knn=cv2.ml.KNearest_create()
knn.train(angle,cv2.ml.ROW_SAMPLE, label)
cap=cv2.VideoCapture(0)

startTime=time.time()
prev_index=0
sentence=''
recognizeDelay=2



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
            
            data=np.array([angle],dtype=np.float32)
            ret, results,neighbouts,dist=knn.findNearest(data,3)
            index=int(results[0][0])
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 26:
                            sentence += ' '
                        elif  index == 27:
                            sentence = ''
                        elif index == 28:
                            sentence = sentence[:-1]  # 문장의 마지막 글자를 삭제합니다.
                        else:
                            sentence += gesture[index]
                        startTime = time.time()
                        
                cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10),
                                          int(res.landmark[0].y * img.shape[0] + 40)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=3)
            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
    cv2.putText(img,sentence,(20,440),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
    
    cv2.imshow('HandTracking',img)
    cv2.waitKey(1)
        
    if keyboard.is_pressed('b'): #  b를 누를시 프로그램 종료
        cap.release()
        cv2.destroyAllWindows()
        break

print(sentence)

import urllib.parse
import urllib.request
import json
from gtts import gTTS
from playsound import playsound
import os

client_id = "nqrRiDBAj1Ubev6cPx4L"
client_secret = "By4tMA1wAE"

enc_text = urllib.parse.quote(sentence)
data = "source=en&target=ko&text=" + enc_text  # 번역 요청 데이터 설정

url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)

response = urllib.request.urlopen(request, data=data.encode('utf-8'))
rescode = response.getcode()

if rescode == 200:
    response_body = response.read()
    res = json.loads(response_body.decode('utf-8'))
    print(res['message']['result']['translatedText']) #res는result의 약자
    translated_text = res['message']['result']['translatedText']  # 번역 결과 추출
    
    tts = gTTS(text=translated_text, lang='ko')  # 번역된 텍스트를 한국어 음성으로 변환
    if os.path.isfile('outputEnglishToKorean.mp3'): os.remove('outputEnglishToKorean.mp3')# 기존 파일삭제
        
    tts.save("outputEnglishToKorean.mp3")  # 음성 파일 저장
    
else:
    print("Error Code:", rescode)

playsound("outputEnglishToKorean.mp3")  # 음성 파일 재생