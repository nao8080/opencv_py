import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # 画像読み込み
    img_bgr = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

    filename = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(filename)  
    faces = face_cascade.detectMultiScale(img_gray, 1.1,5)

    color = (255,0 ,0)
    thickness = 2
    for (x,y,w,h) in faces:
        cv2.rectangle(img_bgr, (x,y), (x + w, y + h),color,thickness)
    
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB));
    
    