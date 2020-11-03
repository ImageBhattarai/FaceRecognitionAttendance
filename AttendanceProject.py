import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Resources'

myList = os.listdir(path)
images = []
students = []
print(myList)

for entry in myList:
    imgCur = cv2.imread(f'{path}/{entry}')
    images.append(imgCur)
    students.append(os.path.splitext(entry)[0])
print(students)

def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encodedList.append(encoded)
    return encodedList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time = datetime.now()
            timeString = time.strftime('%H:%M:%S:')
            f.writelines(f'\n{name},{timeString}')




knownFaces = findEncodings(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(imgS)
    faceEncode = face_recognition.face_encodings(imgS,faceLoc)

    for encoder,location in zip(faceEncode, faceLoc):
        matches = face_recognition.compare_faces(knownFaces, encoder)
        faceDis = face_recognition.face_distance(knownFaces, encoder)
        print(faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = students[matchIndex].upper()
            print(name)
            x1,x2,y1,y2 = location
            x1,x2,y1,y2 = x1*4,x2*4,y1*4,y2*4
            cv2.rectangle(img,(y2,x1),(x2,y1),(0,255,0),2)
            cv2.putText(img,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            markAttendance(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)