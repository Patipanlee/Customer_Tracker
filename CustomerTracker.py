from ultralytics import YOLO
from insightface.app import FaceAnalysis
from collections import defaultdict
import numpy as np
import cv2

class Tracker_Model:
    def __init__(self,yolo="model/yolo11n.pt",insightface="buffalo_s"):
        self.model = YOLO(yolo)
        self.app = FaceAnalysis(name=insightface, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.track_history = defaultdict(lambda: [])

    def customer_tracking(self, frame):
        results = self.model.track(frame, persist=True,classes=[0],tracker='custom.yaml',conf=0.5,iou=0.5)
        return results
    
    def get_face_features(self, frame, bbox,score=0.65):
        person_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        faces = self.app.get(person_crop)
        for face in faces:
            if face.det_score > score:
                gender = "Man" if face.gender == 1 else "Female"
                age = face.age
                return gender, age
        return None, None
    
    def plot_line(self, frame, results):
        boxes = results[0].boxes.xywh.cpu()
        ids = results[0].boxes.id.int().cpu().tolist()
        for box, id in zip(boxes, ids):
            x, y, w, h = box
            track = self.track_history[id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(255, 255, 255), thickness=2)