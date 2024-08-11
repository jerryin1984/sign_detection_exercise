#!/usr/bin/env python3

import rospy
import onnxruntime as ort
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import csv
import time
from collections import Counter
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tf2_msgs.msg import TFMessage

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('roadsign_detector')

class YoloV5:
    def __init__(self, model_path, input_shape=(1, 3, 640, 640)):
        rospy.loginfo(f"Initializing YoloV5 with model: {model_path}")
        try:
            self.ort_session = ort.InferenceSession(model_path)
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            raise
        self.input_shape = input_shape
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        self.mean_val = [0.485, 0.456, 0.406]
        self.scale_val = [0.229, 0.224, 0.225]
        rospy.loginfo("YoloV5 initialized successfully")

    def transform(self, mat_rs):
        canvas = cv2.cvtColor(mat_rs, cv2.COLOR_BGR2RGB)
        canvas = (canvas / 255.0 - self.mean_val) / self.scale_val
        canvas = np.transpose(canvas, (2, 0, 1))  # HWC to CHW
        canvas = np.expand_dims(canvas, axis=0).astype(np.float32)
        return canvas

    def resize_unscale(self, mat, target_height, target_width):
        img_height, img_width = mat.shape[:2]
        r = min(target_width / img_width, target_height / img_height)
        new_unpad_w = int(img_width * r)
        new_unpad_h = int(img_height * r)
        pad_w = target_width - new_unpad_w
        pad_h = target_height - new_unpad_h
        dw = pad_w // 2
        dh = pad_h // 2
        mat_rs = cv2.resize(mat, (new_unpad_w, new_unpad_h))
        mat_rs = cv2.copyMakeBorder(mat_rs, dh, pad_h - dh, dw, pad_w - dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return mat_rs, {'r': r, 'dw': dw, 'dh': dh}

    def detect(self, mat, score_threshold=0.5, iou_threshold=0.4, topk=100, nms_type='hard'):
        rospy.loginfo("Starting detection")
        input_height, input_width = self.input_shape[2], self.input_shape[3]
        img_height, img_width = mat.shape[:2]
        mat_rs, scale_params = self.resize_unscale(mat, input_height, input_width)
        input_tensor = self.transform(mat_rs)
        pred = self.ort_session.run(self.output_names, {self.input_name: input_tensor})[0]
        detected_boxes = self.generate_bboxes(scale_params, pred, score_threshold, img_height, img_width)
        final_boxes = self.nms(detected_boxes, iou_threshold, topk, nms_type)
        rospy.loginfo(f"Detection finished, found {len(final_boxes)} boxes")
        return final_boxes

    def generate_bboxes(self, scale_params, pred, score_threshold, img_height, img_width):
        r = scale_params['r']
        dw = scale_params['dw']
        dh = scale_params['dh']
        bbox_collection = []
        for i in range(pred.shape[1]):
            obj_conf = pred[0, i, 4]
            if obj_conf < score_threshold:
                continue
            cls_conf = pred[0, i, 5:].max()
            label = pred[0, i, 5:].argmax()
            conf = obj_conf * cls_conf
            if conf < score_threshold:
                continue
            cx, cy, w, h = pred[0, i, :4]
            x1 = ((cx - w / 2) - dw) / r
            y1 = ((cy - h / 2) - dh) / r
            x2 = ((cx + w / 2) - dw) / r
            y2 = ((cy + h / 2) - dh) / r
            bbox = {'x1': max(0, x1), 'y1': max(0, y1), 'x2': min(x2, img_width - 1), 'y2': min(y2, img_height - 1), 'score': conf, 'label': label}
            bbox_collection.append(bbox)
        return bbox_collection

    def nms(self, bboxes, iou_threshold, topk, nms_type='hard'):
        bboxes = sorted(bboxes, key=lambda x: x['score'], reverse=True)
        selected_boxes = []
        while bboxes and len(selected_boxes) < topk:
            box = bboxes.pop(0)
            selected_boxes.append(box)
            bboxes = [b for b in bboxes if self.iou(box, b) < iou_threshold]
        return selected_boxes

    @staticmethod
    def iou(box1, box2):
        inter_x1 = max(box1['x1'], box2['x1'])
        inter_y1 = max(box1['y1'], box2['y1'])
        inter_x2 = min(box1['x2'], box2['x2'])
        inter_y2 = min(box1['y2'], box2['y2'])
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        return inter_area / (box1_area + box2_area - inter_area)

class RoadsSignDetector:
    def __init__(self):
        rospy.init_node('roadsign_detector', anonymous=False)

        # 在程序开始时获取地名输入
        self.target_city = input("ZFJ needs you to type your destination(:：")

        model1_path = rospy.get_param('~model1_path', '/home/kal4/kal43_ws/src/roadsign_detector/src/model/model1.onnx')
        model2_path = rospy.get_param('~model2_path', '/home/kal4/kal43_ws/src/roadsign_detector/src/model/model2.onnx')
        self.depth_scale = rospy.get_param('~depth_scale', 0.0001)
        self.csv_file = rospy.get_param('~csv_file', '/home/kal4/kal43_ws/src/roadsign_detector/src/road_signs_info.csv')
        self.x_min = rospy.get_param('~x_min', 0.25)
        self.x_max = rospy.get_param('~x_max', 8.8)
        self.y_min = rospy.get_param('~y_min', 1.7)
        self.y_max = rospy.get_param('~y_max', 5.0)

        rospy.loginfo(f"Parameters: model1_path={model1_path}, model2_path={model2_path}, depth_scale={self.depth_scale}, csv_file={self.csv_file}, x_min={self.x_min}, x_max={self.x_max}, y_min={self.y_min}, y_max={self.y_max}, target_city={self.target_city}")
        
        try:
            self.sign_detector = YoloV5(model1_path)
            self.city_detector = YoloV5(model2_path, input_shape=(1, 3, 224, 224))
        except Exception as e:
            rospy.logerr(f"Failed to initialize detectors: {e}")
            raise

        self.bridge = CvBridge()
        self.pub_realtime = rospy.Publisher('/zfj/realtime_info', String, queue_size=10)
        self.pub_accurate = rospy.Publisher('/zfj/accurate_info', String, queue_size=10)
        self.image_sub = None
        self.depth_sub = None
        self.lock = threading.Lock()
        self.depth_image = None
        self.initialize_csv_file()
        self.image_queue = queue.Queue(maxsize=1000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.start_csv_check_thread()
        self.start_image_processing_threads()
        
        # 订阅tf话题，接收变换信息
        self.tf_subscriber = rospy.Subscriber('/tf', TFMessage, self.tf_callback)
        
        rospy.loginfo("RoadsSignDetector initialized.")
        rospy.loginfo("Subscribed to /tf")

    # tf消息的回调函数
    def tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "camera_top":
                translation_x = transform.transform.translation.x
                translation_y = transform.transform.translation.y

                rospy.loginfo(f"Received /tf message: x={translation_x}, y={translation_y}")

                # 检查translation的值是否在设定的区间内
                if self.x_min <= translation_x <= self.x_max and self.y_min <= translation_y <= self.y_max:
                    rospy.loginfo(f"Translation within limits: x={translation_x}, y={translation_y}")
                    # 如果在区间内，订阅图像话题
                    if self.image_sub is None:
                        self.image_sub = rospy.Subscriber('/camera_front/color/image_raw', Image, self.image_callback)
                        rospy.loginfo("Subscribed to /camera_front/color/image_raw")
                    if self.depth_sub is None:
                        self.depth_sub = rospy.Subscriber('/camera_front/depth/image_rect_raw', Image, self.depth_callback)
                        rospy.loginfo("Subscribed to /camera_front/depth/image_rect_raw")
                else:
                    rospy.loginfo(f"Translation not within limits: x={translation_x}, y={translation_y}")
                    # 如果不在区间内，取消订阅图像话题
                    if self.image_sub is not None:
                        self.image_sub.unregister()
                        self.image_sub = None
                        rospy.loginfo("Unsubscribed from /camera_front/color/image_raw")
                    if self.depth_sub is not None:
                        self.depth_sub.unregister()
                        self.depth_sub = None
                        rospy.loginfo("Unsubscribed from /camera_front/depth/image_rect_raw")

    def initialize_csv_file(self):
        rospy.loginfo(f"Initializing CSV file at {self.csv_file}")
        with open(self.csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'City', 'Direction', 'Distance'])

    def image_callback(self, data):
        rospy.loginfo("Image callback triggered")
        try:
            cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if not self.image_queue.full():
                self.image_queue.put(cv_img)
                rospy.loginfo("Image added to queue")
            else:
                rospy.logwarn("Image queue is full, skipping frame")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def depth_callback(self, data):
        rospy.loginfo("Depth callback triggered")
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            rospy.loginfo("Depth image received")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def start_image_processing_threads(self):
        rospy.loginfo("Starting image processing threads")
        for _ in range(4):
            thread = threading.Thread(target=self.process_images_from_queue)
            thread.daemon = True
            thread.start()

    def process_images_from_queue(self):
        while not rospy.is_shutdown():
            try:
                cv_img = self.image_queue.get()
                self.process_image(cv_img)
                self.image_queue.task_done()
            except Exception as e:
                rospy.logerr(f"Error in process_images_from_queue: {e}")

    def process_image(self, cv_img):
        rospy.loginfo("Processing image")
        try:
            detected_boxes = self.sign_detector.detect(cv_img, score_threshold=0.75)
            futures = []
            for box in detected_boxes:
                label = 'left' if box['label'] == 0 else 'right'
                cropped_img = cv_img[int(box['y1']):int(box['y2']), int(box['x1']):int(box['x2'])]
                futures.append(self.executor.submit(self.process_cropped_image, cropped_img, label, box))
            for future in as_completed(futures):
                future.result()
        except Exception as e:
            rospy.logerr(f"Error in process_image: {e}")

    def process_cropped_image(self, cropped_img, label, box):
        rospy.loginfo(f"Processing cropped image for label: {label}, box: {box}")
        try:
            resized_img = cv2.resize(cropped_img, (224, 224))
            city_boxes = self.city_detector.detect(resized_img, score_threshold=0.75)
            distance = self.calculate_distance(box) if self.depth_image is not None else -1
            for city_box in city_boxes:
                city = self.get_city_name(city_box['label'])
                if city == self.target_city:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                    info = f"{timestamp}, {city}, {label}, {distance:.2f}m"
                    rospy.loginfo(info)
                    self.pub_realtime.publish(info)
                    with self.lock:
                        with open(self.csv_file, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, city, label, f"{distance:.2f}m"])
        except Exception as e:
            rospy.logerr(f"Error in process_cropped_image: {e}")

    def calculate_distance(self, box):
        rospy.loginfo("Calculating distance")
        try:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            depth_region = self.depth_image[y1:y2, x1:x2]
            valid_depths = depth_region[depth_region > 0]
            if valid_depths.size > 0:
                distance = np.median(valid_depths) * self.depth_scale  # Convert depth to meters
                rospy.loginfo(f"Calculated distance: {distance:.2f}m")
                return distance
            return -1
        except Exception as e:
            rospy.logerr(f"Error in calculate_distance: {e}")
            return -1

    def get_city_name(self, label):
        city_names = {0: 'hildesheim', 1: 'karlsruhe', 2: 'koln', 3: 'munchen'}
        city_name = city_names.get(label, 'unknown')
        rospy.loginfo(f"Detected city: {city_name}")
        return city_name

    def start_csv_check_thread(self):
        rospy.loginfo("Starting CSV check thread")
        thread = threading.Thread(target=self.check_csv_file)
        thread.daemon = True
        thread.start()

    def check_csv_file(self):
        while not rospy.is_shutdown():
            with self.lock:
                with open(self.csv_file, 'r') as f:
                    reader = csv.reader(f)
                    data = list(reader)[1:]  # Skip header row
                    if data:
                        city_directions = self.get_most_common_directions(data)
                        filtered_directions = {city: direction for city, direction in city_directions.items() if city == self.target_city}
                        info = ', '.join([f"{city}: {direction}" for city, direction in filtered_directions.items()])
                        rospy.loginfo(f"Publishing accurate info: {info}")
                        self.pub_accurate.publish(info)
            time.sleep(0.001)

    def get_most_common_directions(self, data):
        direction_counts = {city: Counter() for city in ['hildesheim', 'karlsruhe', 'koln', 'munchen']}
        for row in data:
            city = row[1]
            direction = row[2]
            direction_counts[city][direction] += 1
        most_common_directions = {city: direction_count.most_common(1)[0][0] if direction_count else 'unknown' for city, direction_count in direction_counts.items()}
        return most_common_directions

if __name__ == "__main__":
    try:
        detector = RoadsSignDetector()
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"Unhandled exception: {e}")
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
