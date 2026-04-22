import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import time

# ====== YOLO Detector ======
class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.55, iou_threshold=0.7):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = 320

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def postprocess(self, outputs, frame_shape):
        predictions = outputs[0].transpose(0, 2, 1)
        boxes, confidences = [], []
        h, w = frame_shape[:2]
        scale_x = w / self.input_size
        scale_y = h / self.input_size

        for pred in predictions[0]:
            x_center, y_center, width, height, confidence = pred
            if confidence > self.conf_threshold:
                x_center *= scale_x
                y_center *= scale_y
                bw = width * scale_x
                bh = height * scale_y
                x1 = max(0, min(int(x_center - bw / 2), w))
                y1 = max(0, min(int(y_center - bh / 2), h))
                bw = min(int(bw), w - x1)
                bh = min(int(bh), h - y1)
                boxes.append([x1, y1, bw, bh])
                confidences.append(float(confidence))

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes      = [boxes[i] for i in indices]
                confidences = [confidences[i] for i in indices]

        if len(boxes) > 1:
            idx = int(np.argmax(confidences))
            boxes       = [boxes[idx]]
            confidences = [confidences[idx]]

        return boxes, confidences

    def detect(self, frame):
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(outputs, frame.shape)


# ====== MobileNetV2 Classifier ======
class MobileNetClassifier:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size  = 224
        self.class_names = ['def_front', 'ok_front']
        self.colors      = [(0, 0, 255), (0, 255, 0)]

    def classify(self, image):
        # CLAHE cân bằng sáng
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        pred_class = int(np.argmax(output[0]))
        confidence = float(output[0][pred_class])
        return pred_class, confidence


# ====== Load models ======
print("Loading models...")
yolo = YOLODetector(r"best.onnx")
classifier = MobileNetClassifier(r"mobilenetv2_optimized.tflite")

print("Models loaded\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ====== Config ======
DET_TRIGGER   = 0.80   
STABLE_SEC    = 2.0    # Chờ 2s để ảnh ổn định trước classify
COOLDOWN_SEC  = 2.0    # Chờ 2s sau classify để đổi sản phẩm

# ====== States ======
STATE_WAITING     = "WAITING"       
STATE_STABILIZING = "STABILIZING"   
STATE_CLASSIFIED  = "CLASSIFIED"    
STATE_COOLDOWN    = "COOLDOWN"      

state            = STATE_WAITING
state_start_time = None

# ====== Counter ======
count_ok        = 0
count_def       = 0
last_pred_class = None
last_class_conf = 0.0
last_det_conf   = 0.0
last_box        = None

prev_time = time.time()
print("Press q=quit | r=reset\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # YOLO detect toàn frame
    boxes, det_confidences = yolo.detect(frame)

    # Lấy detection tốt nhất (nếu có)
    best_box  = boxes[0]          if len(boxes) > 0 else None
    best_conf = det_confidences[0] if len(boxes) > 0 else 0.0

    # ====== STATE MACHINE ======
    if state == STATE_WAITING:
        # Trigger khi YOLO detect >= 80%
        if best_box is not None and best_conf >= DET_TRIGGER:
            state = STATE_STABILIZING
            state_start_time = now
            print(f"YOLO {best_conf:.0%} >= 80% → đang ổn định 2s...")

    elif state == STATE_STABILIZING:
        elapsed = now - state_start_time
        if best_box is None or best_conf < DET_TRIGGER:
            # Mất detection → reset, chờ lại
            state = STATE_WAITING
            state_start_time = None
            print("Mất detection, reset.")
        elif elapsed >= STABLE_SEC:
            # Đủ 2s + detection ổn định → classify
            x, y, w, h = best_box
            roi = frame[y:y+h, x:x+w]
            if roi.shape[0] >= 20 and roi.shape[1] >= 20:
                pred_class, class_conf = classifier.classify(roi)
                last_pred_class = pred_class
                last_class_conf = class_conf
                last_det_conf   = best_conf
                last_box        = best_box

                if pred_class == 0:
                    count_def += 1
                    print(f"[DEF] #{count_def} conf={class_conf:.2f} | OK={count_ok} DEF={count_def}")
                else:
                    count_ok += 1
                    print(f"[OK]  #{count_ok} conf={class_conf:.2f} | OK={count_ok} DEF={count_def}")

                state = STATE_CLASSIFIED

    elif state == STATE_CLASSIFIED:
        # Chuyển ngay sang cooldown
        state = STATE_COOLDOWN
        state_start_time = now

    elif state == STATE_COOLDOWN:
        elapsed = now - state_start_time
        if elapsed >= COOLDOWN_SEC:
            state = STATE_WAITING
            state_start_time = None
            print("Sẵn sàng ảnh tiếp theo...\n")

    # ====== Vẽ bounding box ======
    draw_box  = best_box  if state in (STATE_WAITING, STATE_STABILIZING) else last_box
    draw_conf = best_conf if state in (STATE_WAITING, STATE_STABILIZING) else last_det_conf

    if draw_box is not None:
        x, y, w, h = draw_box
        if state == STATE_WAITING:
            box_color = (128, 128, 128)   # Xám: chưa đủ confidence
        elif state == STATE_STABILIZING:
            box_color = (0, 255, 255)     # Vàng: đang ổn định
        else:
            box_color = classifier.colors[last_pred_class] if last_pred_class is not None else (200,200,200)

        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 3)

        # Label confidence YOLO
        det_label = f"Det: {draw_conf:.0%}"
        cv2.putText(frame, det_label, (x+5, y+h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # Label kết quả classify (chỉ khi đã classify)
        if state in (STATE_CLASSIFIED, STATE_COOLDOWN) and last_pred_class is not None:
            result_label = f"{classifier.class_names[last_pred_class]}: {last_class_conf:.2f}"
            (tw, th), bl = cv2.getTextSize(result_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y-th-bl-15), (x+tw+10, y), box_color, -1)
            cv2.putText(frame, result_label, (x+5, y-bl-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # ====== Status bar (top-left) ======
    if state == STATE_WAITING:
        status = f"WAITING... (need YOLO >= {DET_TRIGGER:.0%})"
        s_color = (255, 165, 0)
        progress = 0.0

    elif state == STATE_STABILIZING:
        elapsed  = now - state_start_time
        progress = min(elapsed / STABLE_SEC, 1.0)
        s_color  = (0, 255, 255)
        status   = f"STABILIZING {STABLE_SEC - elapsed:.1f}s  |  Det: {best_conf:.0%}"

    elif state == STATE_CLASSIFIED:
        s_color  = classifier.colors[last_pred_class]
        status   = f">> {classifier.class_names[last_pred_class].upper()} ({last_class_conf:.2f}) <<"
        progress = 1.0

    elif state == STATE_COOLDOWN:
        elapsed  = now - state_start_time
        progress = 1.0 - min(elapsed / COOLDOWN_SEC, 1.0)
        s_color  = classifier.colors[last_pred_class]
        status   = f"{classifier.class_names[last_pred_class].upper()} | Change product {COOLDOWN_SEC-elapsed:.1f}s"

    # Thanh trạng thái ngang trên cùng
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
    cv2.putText(frame, status, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, s_color, 2)

    # Progress bar bên dưới status bar
    bar_w = frame.shape[1] - 20
    filled = int(bar_w * progress)
    cv2.rectangle(frame, (10, 52), (10+bar_w, 62), (60,60,60), -1)
    if filled > 0:
        cv2.rectangle(frame, (10, 52), (10+filled, 62), s_color, -1)

    # ====== FPS ======
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS:{fps:.1f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # ====== Counter Panel (top-right) ======
    px = frame.shape[1] - 290
    cv2.rectangle(frame, (px-10, 68), (frame.shape[1]-8, 175), (30,30,30), -1)
    cv2.rectangle(frame, (px-10, 68), (frame.shape[1]-8, 175), (100,100,100), 1)
    cv2.putText(frame, f"OK   : {count_ok}",  (px, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(frame, f"DEF  : {count_def}", (px, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.putText(frame, f"TOTAL: {count_ok+count_def}", (px, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # ====== Legend ======
    cv2.putText(frame, "q=quit | r=reset counter",
                (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    cv2.imshow('Product Classification', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        count_ok = 0
        count_def = 0
        last_pred_class = None
        last_box = None
        state = STATE_WAITING
        state_start_time = None
        print("\nCounter reset!\n")

cap.release()
cv2.destroyAllWindows()
print(f"\n=== KẾT QUẢ ===")
print(f"OK  : {count_ok}")
print(f"DEF : {count_def}")
print(f"Tổng: {count_ok + count_def}")
