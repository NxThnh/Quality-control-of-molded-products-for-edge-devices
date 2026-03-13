import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import gradio as gr

# ========================================================
# ====== CẤU HÌNH NGƯỠNG TÙY CHỈNH (CHỈNH SỬA Ở ĐÂY) ======
# ========================================================
CONFIDENCE_THRESHOLD = 0.55  # Ngưỡng độ tự tin để nhận diện phôi
IOU_THRESHOLD = 0.50           # Ngưỡng chống trùng lặp hộp (IoU)
COUNTING_LINE_X = 320          # TỌA ĐỘ X CỦA VẠCH ĐẾM DỌC (Tùy chỉnh theo camera thực tế)
# ========================================================

# ====== THUẬT TOÁN THEO DÕI VẬT THỂ (TRACKING) ======
class SimpleTracker:
    def __init__(self, max_distance=150, max_lost=5):
        self.max_distance = max_distance 
        self.max_lost = max_lost   
        self.reset() 

    def reset(self):
        self.next_id = 1
        self.tracked_objects = {}  
        self.history = []          
        self.class_records = {} 
        self.counted_ids = set() # Set để lưu các ID đã đi qua vạch

    def update(self, detections):
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['lost'] += 1
            
        current_frame_objects = {}
        
        for det in detections:
            cx, cy = det['cx'], det['cy']
            closest_id = None
            min_dist = self.max_distance
            
            for obj_id, obj_data in self.tracked_objects.items():
                dist = ((cx - obj_data['cx'])**2 + (cy - obj_data['cy'])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_id = obj_id
                    
            if closest_id is not None:
                det['lost'] = 0
                prev_cx = self.tracked_objects[closest_id]['cx'] # Lưu lại X của frame trước
                curr_cx = det['cx']
                
                self.tracked_objects[closest_id] = det
                current_frame_objects[closest_id] = det
                
                # LOGIC VƯỢT VẠCH ĐỂ ĐẾM (DI CHUYỂN TỪ TRÁI SANG PHẢI)
                # Nếu frame trước ở bên trái vạch (< LINE) và frame này ở bên phải/chạm vạch (>= LINE)
                if prev_cx < COUNTING_LINE_X and curr_cx >= COUNTING_LINE_X:
                    if closest_id not in self.counted_ids:
                        self.counted_ids.add(closest_id) 
                        
                        self.class_records[closest_id] = det['class_id']
                        
                        new_hist = {'id': closest_id}
                        new_hist.update(det)
                        self.history.insert(0, new_hist)
                        if len(self.history) > 15: 
                            self.history.pop()
            else:
                obj_id = self.next_id
                self.next_id += 1
                det['lost'] = 0
                self.tracked_objects[obj_id] = det
                current_frame_objects[obj_id] = det
                    
        to_delete = [obj_id for obj_id, data in self.tracked_objects.items() if data['lost'] > self.max_lost]
        for obj_id in to_delete:
            del self.tracked_objects[obj_id]
            
        return current_frame_objects

# ====== GIAI ĐOẠN 1: YOLO DETECTOR ======
class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.55, iou_threshold=0.75):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = 640
        
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
            conf = pred[4]
            if conf > self.conf_threshold:
                cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
                xmin = int((cx - bw/2) * scale_x)
                ymin = int((cy - bh/2) * scale_y)
                xmax = int((cx + bw/2) * scale_x)
                ymax = int((cy + bh/2) * scale_y)
                boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                confidences.append(float(conf))
                
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

# ====== GIAI ĐOẠN 2: MOBILENET CLASSIFIER ======
class MobileNetClassifier:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.class_names = ['ok_front', 'def_front']
        self.colors = [(50, 255, 50), (50, 50, 255)] 
        self.hex_colors = ["#32FF32", "#FF3232"]
        
    def predict(self, crop_img):
        img = cv2.resize(crop_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return np.argmax(output)

# Khởi tạo Hệ thống
detector = YOLODetector("best.onnx", conf_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD)
classifier = MobileNetClassifier("mobilenetv2_optimized.tflite")
tracker = SimpleTracker(max_distance=150)

# ====== HÀM TẠO GIAO DIỆN BẢNG (HTML) ======
def generate_dashboard_html():
    total_count = len(tracker.class_records)
    total_ok = sum(1 for cid in tracker.class_records.values() if cid == 0)
    total_def = sum(1 for cid in tracker.class_records.values() if cid == 1)

    html_content = f"""
    <div style='font-family: sans-serif;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 20px;'>
            <div style='background: #e9ecef; padding: 10px 0; border-radius: 8px; width: 31%; text-align: center; border: 1px solid #ced4da;'>
                <b style='color: #495057; font-size: 14px;'>TỔNG SỐ</b><br>
                <span style='font-size: 28px; font-weight: bold; color: #212529;'>{total_count}</span>
            </div>
            <div style='background: #d4edda; padding: 10px 0; border-radius: 8px; width: 31%; text-align: center; border: 1px solid #c3e6cb;'>
                <b style='color: #155724; font-size: 14px;'>ĐẠT (OK)</b><br>
                <span style='font-size: 28px; font-weight: bold; color: #28a745;'>{total_ok}</span>
            </div>
            <div style='background: #f8d7da; padding: 10px 0; border-radius: 8px; width: 31%; text-align: center; border: 1px solid #f5c6cb;'>
                <b style='color: #721c24; font-size: 14px;'>LỖI (DEFECT)</b><br>
                <span style='font-size: 28px; font-weight: bold; color: #dc3545;'>{total_def}</span>
            </div>
        </div>

        <table style='width: 100%; border-collapse: collapse; text-align: left;'>
            <tr style='background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;'>
                <th style='padding: 10px;'>Mã Phôi (ID)</th>
                <th style='padding: 10px; text-align: right;'>Kết Luận</th>
            </tr>
    """
    
    for item in tracker.history:
        c_id = item['class_id']
        html_color = classifier.hex_colors[c_id]
        display_label = "ĐẠT" if c_id == 0 else "LỖI"
        bg_color = "rgba(50, 255, 50, 0.05)" if c_id == 0 else "rgba(255, 50, 50, 0.05)"
        
        html_content += f"""
            <tr style='border-bottom: 1px solid #e9ecef; background-color: {bg_color};'>
                <td style='padding: 10px; font-weight: bold; color: #495057;'>#{item['id']}</td>
                <td style='padding: 10px; text-align: right; color: {html_color}; font-weight: bold;'>{display_label}</td>
            </tr>
        """
    html_content += "</table></div>"

    if not tracker.history:
        html_content += "<div style='color: gray; padding: 10px; text-align: center;'>Chưa có phôi nào vượt qua vạch đếm...</div>"
        
    return html_content

# ====== HÀM XỬ LÝ FRAME CAMERA ======
def process_frame(rgb_frame):
    if rgb_frame is None:
        return None, generate_dashboard_html()
    
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # YOLO Phát hiện
    input_tensor = detector.preprocess(frame)
    outputs = detector.session.run(None, {detector.input_name: input_tensor})
    boxes = detector.postprocess(outputs, frame.shape)
    
    detections = []
    
    # Phân loại và gom dữ liệu
    for box in boxes:
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0: continue
            
        crop = frame[y:y+h, x:x+w]
        class_id = classifier.predict(crop)
        
        detections.append({
            'box': (x, y, w, h),
            'cx': x + w//2,
            'cy': y + h//2,
            'class_id': class_id
        })
        
    # Cập nhật Tracker
    active_objects = tracker.update(detections)
    
    # VẼ VẠCH RANH GIỚI DỌC MÀU VÀNG LÊN CAMERA
    cv2.line(frame, (COUNTING_LINE_X, 0), (COUNTING_LINE_X, frame.shape[0]), (0, 255, 255), 2)
    cv2.putText(frame, "Counting Line", (COUNTING_LINE_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Vẽ lên Camera
    for obj_id, data in active_objects.items():
        x, y, w, h = data['box']
        c_id = data['class_id']
        label = classifier.class_names[c_id]
        cv_color = classifier.colors[c_id]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), cv_color, 2)
        cv2.circle(frame, (data['cx'], data['cy']), 4, cv_color, -1)
        
        display_text = f"#{obj_id} {label}"
        cv2.putText(frame, display_text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cv_color, 2)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), generate_dashboard_html()

# ====== HÀM NÚT BẤM RESET ======
def reset_system():
    tracker.reset()
    return generate_dashboard_html()

# ====== GIAO DIỆN WEB ======
with gr.Blocks(title="Casting Defect Inspection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("### Hệ thống phân loại lỗi phôi đúc")
    
    with gr.Row():
        with gr.Column(scale=7):
            input_video = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Gốc")
            output_video = gr.Image(type="numpy", label="Màn hình giám sát AI")
            
        with gr.Column(scale=3):
            gr.Markdown("**Bảng Thống Kê & Nhật Ký:**")
            prediction_panel = gr.HTML(value=generate_dashboard_html())
            # Nút bấm Reset
            reset_btn = gr.Button("🔄 Reset Bộ Đếm", variant="secondary")
    
    # Luồng xử lý Camera
    input_video.stream(
        fn=process_frame, 
        inputs=input_video, 
        outputs=[output_video, prediction_panel], 
        stream_every=0.1
    )
    
    # Xử lý sự kiện bấm nút Reset (Cập nhật lại bảng HTML)
    reset_btn.click(fn=reset_system, outputs=prediction_panel)

if __name__ == "__main__":
    demo.launch()