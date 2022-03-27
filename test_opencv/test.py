#pip install opencv-python
import cv2

#노트북 카메라에서 영상을 읽어온다
cap = cv2.VideoCapture(0)

#얼굴 인식 캐스케이드 파일 읽는다
face_cascade = cv2.CascadeClassifier('C:/Users/srk99/Downloads/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')

while(True):
    # frame 별로 capture 한다
    ret, frame = cap.read()
	
    # 좌우 반전은 1, 상하반전은 0
    frame = cv2.flip(frame,1)
	# 프레임이 제대로 읽어지지 않은 경우
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detectMultiScale (InputArray image, std::vector< Rect > &objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)

    # 빨간 사각형으로 인식된 얼굴을 표시한다.
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    #webCamera라는 이름으로 실시간 화면을 보여준다.
    cv2.imshow('webCamera',frame)
    # q를 누르면 종료되도록 하는 코드이다.
    if cv2.waitKey(1) == ord('q'):
        break
        
# 메모리를 해제시켜준다.
cap.release()
cv2.destroyAllWindows()