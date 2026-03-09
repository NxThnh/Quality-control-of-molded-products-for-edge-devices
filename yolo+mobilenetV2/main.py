import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf

# ====== YOLO Detector (Phase 1: Detection) ======
class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.55, iou_threshold=0.6):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = 320
        
    def preprocess(self, frame):
        self.original_shape = frame.shape[:2]
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    
    def postprocess(self, outputs, frame_shape):
        predictions = outputs[0].transpose(0, 2, 1)
        
        boxes = []
        confidences = []
        
        h, w = frame_shape[:2]
        scale_x = w / self.input_size
        scale_y = h / self.input_size
        
        for pred in predictions[0]:
            x_center, y_center, width, height, confidence = pred
            
            if confidence > self.conf_threshold:
                x_center = x_center * scale_x
                y_center = y_center * scale_y
                box_width = width * scale_x
                box_height = height * scale_y
                
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                box_width = min(int(box_width), w - x1)
                box_height = min(int(box_height), h - y1)
                
                boxes.append([x1, y1, box_width, box_height])
                confidences.append(float(confidence))
        
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                confidences = [confidences[i] for i in indices]
                
                # Chỉ giữ 1 detection có confidence cao nhất
                if len(boxes) > 1:
                    max_conf_idx = np.argmax(confidences)
                    boxes = [boxes[max_conf_idx]]
                    confidences = [confidences[max_conf_idx]]
        
        return boxes, confidences
    
    def detect(self, frame):
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes, confidences = self.postprocess(outputs, frame.shape)
        return boxes, confidences


# ====== MobileNetV2 Classifier (Phase 2: Classification) ======
class MobileNetClassifier:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = 224
        
        # Class names: 0 = def_front, 1 = ok_front
        self.class_names = ['def_front', 'ok_front']
        self.colors = [(0, 0, 255), (0, 255, 0)]  # Đỏ cho def, Xanh cho ok
        
    def classify(self, image):
        """Classify cropped image"""
        # Preprocess
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get prediction
        pred_class = np.argmax(output[0])
        confidence = output[0][pred_class]
        
        return pred_class, confidence


# ====== Main Application ======
print("Loading models...")
yolo = YOLODetector(r"yolo+mobilenetV2\best.onnx", conf_threshold=0.55)
classifier = MobileNetClassifier(r"yolo+mobilenetV2\mobilenetv2_optimized.tflite")
print("✓ Models loaded successfully\n")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting detection...")
print("Press 'q' to quit\n")

import time
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # PHASE 1: Detect objects với YOLO
    boxes, det_confidences = yolo.detect(frame)
    
    # PHASE 2: Classify từng detected object
    for box, det_conf in zip(boxes, det_confidences):
        x, y, w, h = box
        
        # Crop ROI
        roi = frame[y:y+h, x:x+w]
        
        # Skip nếu ROI quá nhỏ
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            continue
        
        # Classify
        pred_class, class_conf = classifier.classify(roi)
        class_name = classifier.class_names[pred_class]
        color = classifier.colors[pred_class]
        
        # Tạo label
        label = f"{class_name}: {class_conf:.2f}"
        det_label = f"Det: {det_conf:.2f}"
        
        # Vẽ bounding box (màu đỏ nếu def, xanh nếu ok)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Vẽ label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y-text_h-baseline-15), (x+text_w+10, y), color, -1)
        
        # Vẽ class label
        cv2.putText(frame, label, (x+5, y-baseline-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Vẽ detection confidence (nhỏ hơn, ở góc dưới box)
        cv2.putText(frame, det_label, (x+5, y+h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Vẽ center point
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Display info
    info = f"FPS: {fps:.1f} | Detections: {len(boxes)}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Hiển thị legend
    cv2.putText(frame, "RED = def_front | GREEN = ok_front", (10, frame.shape[0]-50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('Casting Detection & Classification', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDetection stopped")
