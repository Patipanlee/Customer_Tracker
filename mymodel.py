from CustomerTracker import Tracker_Model
import cv2

my_model = Tracker_Model() # yolo="yolo11n.pt" insightface="buffalo_l"
path = 0 # for local cam
frame_id = 0
cap = cv2.VideoCapture(path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    data = my_model.customer_tracking(frame)
    frame_id += 1
    if data[0].boxes.id is None:
        continue
    track_id = data[0].boxes.id.cpu().numpy()
    bboxs = data[0].boxes.xyxy.cpu().numpy()
    for i in range(len(track_id)):
        x1, y1, x2, y2 = bboxs[i]
        id = track_id[i]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {int(id)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        try:
            gender, age = my_model.get_face_features(frame, [x1, y1, x2, y2])
            cv2.putText(frame, f"Gender: {gender}", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Age: {age}", (int(x1), int(y1) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"Frame ID: {frame_id}, ID: {int(id)}, Gender: {gender}, Age: {age}")
        except:
            cv2.putText(frame, "No Face Detected", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Frame ID: {frame_id}, ID: {int(id)}, No Face Detected")
        my_model.plot_line(frame, data)
    cv2.imshow("YOLOv11 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()