import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime

video_capture=cv2.VideoCapture(0)


#Load Known faces to compare
akshay_image=face_recognition.load_image_file("faces/akshay.jpg")
akshay_encoding=face_recognition.face_encodings(akshay_image)[0]

salman_image=face_recognition.load_image_file("faces/salman.jpg")
salman_encoding=face_recognition.face_encodings(salman_image)[0]

known_faces_encodings= [akshay_encoding,salman_encoding]
known_faces_names=["Akshay","Salman"]

#list of expected students
students=known_faces_names.copy()

face_locations=[]
face_encodings=[]

#Get the current Date and Time
now=datetime.now()
current_date=now.strftime("%d-%m-%y")

f=open(f"{current_date}.csv","w+",newline="")
lnwriter=csv.writer(f)

while True:
    _, frame= video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #Recognize Faces
    face_locations=face_recognition.face_locations(rgb_small_frame)
    face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        matches=face_recognition.compare_faces(known_faces_encodings,face_encoding)
        face_distance=face_recognition.face_distance(known_faces_encodings,face_encoding)
        best_match_index=np.argmin(face_distance)
        if(matches[best_match_index]):
            name =known_faces_names[best_match_index]
            #Add the text if person is recognized
        if name in known_faces_names:
            font=cv2.FONT_HERSHEY_COMPLEX
            bottomLeftCornerOfText=(10,100)
            fontScale=1.0
            fontColor=(255,0,0)
            thickness=3
            lineType=2
            cv2.putText(frame, name +" is Present",bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)
            if name in students:
                students.remove(name)
                current_time=now.strftime("%H-%M")
                lnwriter.writerow([name , current_time])

    
    cv2.imshow("Attendence",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()



