from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
points = []
dragging = False
selected_point = None
removing = False
def find_nearest_point(x, y, threshold=10):
    """Tìm điểm gần nhất với tọa độ (x, y) trong danh sách points"""
    for i, (px, py) in enumerate(points):
        if abs(px - x) < threshold and abs(py - y) < threshold:
            return i
    return None
def click(event, x, y, flags, param):
    global points, dragging, selected_point, removing
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = find_nearest_point(x, y)
        if selected_point is None:
            points.append((x,y))
        else:
            dragging = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_point is not None:
            points[selected_point] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_point = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_point = find_nearest_point(x, y)
        if selected_point is not None:
            removing = True
    elif event == cv2.EVENT_RBUTTONUP:
        if removing and selected_point is not None:
            del points[selected_point]

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../videos/cars.mp4")
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO('yolo_weights/yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# mask = cv2.imread("mask.png")
# mask = cv2.resize(mask,(1280,720))
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
limits = [230, 400, 673, 400]
totalCount = []
success, img = cap.read()
mask = np.zeros_like(img, dtype=np.uint8)
if success:
    while True:
        img_mask = img.copy()
        totalPoint = len(points)
        if totalPoint > 2:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 225))  # white polygon
            pass
        for i in range(totalPoint):
            cv2.circle(img_mask, points[i], 2, (0, 255, 0), -1)
            if totalPoint > 2:
                cv2.line(img_mask, points[i % totalPoint], points[(i + 1) % totalPoint], (0, 255, 0), 2)
                pass
            pass
        cv2.line(img_mask, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        cv2.imshow("Image", img_mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            break

while cap.isOpened():
    success, img = cap.read()
    imgGraphic = cv2.imread('graphics.png',cv2.IMREAD_UNCHANGED)
    for i in range(totalPoint):
        cv2.circle(img, points[i], 2, (0, 255, 0), -1)
        if totalPoint > 2:
            cv2.line(img, points[i % totalPoint], points[(i + 1) % totalPoint], (0, 255, 0), 2)
            pass
        pass
    imRegion = cv2.bitwise_and(img, mask)
    img = cvzone.overlayPNG(img, imgGraphic, (0, 0))
    results = model(imRegion, stream=True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # x1, y1, w, h = box.xywh[0]
            # x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            w,h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            # confident
            conf = math.ceil((box.conf[0]*100))/100
            # class
            cls = box.cls[0]
            currentClass = classNames[int(cls)]
            if currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus' or currentClass == 'motorbike'\
                and conf>0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections,currentArray))
    resultTrackers = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255),5)
    for result in resultTrackers:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255,0, 255), l=8)
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(0, y1)), scale=2,
                           thickness=1, offset=5)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy), radius=5, color=(255,0,255), thickness=cv2.FILLED)
        if limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[1]+15:
            if totalCount.count(id)==0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    # cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, color=(50,50,255), thickness=8)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()


