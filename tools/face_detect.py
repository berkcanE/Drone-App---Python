import cv2 as cv ## importing the opencv

capture = cv.VideoCapture(0)# accessing the camera (index 0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')



while True:
    ret, frame = capture.read()
    gray_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale  , 1.3, 5)
    print(len(faces))

    for(x, y, width, height) in faces: # the coordinate of the face(x,y,width,height)
        cv.rectangle(frame, (x,y), (x+width, y+height), (255, 0, 0), 2)
        eye_gray = gray_scale[y:y+height, x:x+width]
        eye_color = frame[y:y + height, x:x+width]
        eyes = eye_cascade.detectMultiScale(eye_gray)
        for(ex, ey, ew, eh) in eyes:
            cv.rectangle(eye_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

