from person_detection_main import Model
import cv2


cap = cv2.VideoCapture('vid.mp4')

# pass m or l in model to load yolov5m or yolov5l
obj = Model(model = 'l' , confidence = 0.15 , iou  = 0.45)

while True:

    ret, frame = cap.read()
    dict, img = obj.run(frame)

    cv2.imshow('a', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
