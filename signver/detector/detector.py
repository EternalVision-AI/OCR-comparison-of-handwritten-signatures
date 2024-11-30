# import tensorflow as tf
import cv2
import numpy as np
import time

def image_preprocessing(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.threshold(img_gray, thresh=220, maxval=255, type=cv2.THRESH_BINARY)[1]
    return im_bw

class Detector():
    def __init__(self) -> None:
        self.model_load_time = None
        self.model = None
        self.INPUT_WIDTH = 640
        self.CLASSES = ['sign', 'date', 'chk', 'chkarea']

    def load(self, model_path: str) -> None:
        start_time = time.time()
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.model_load_time = time.time() - start_time

    def detect(self, image_path):
        original_image = cv2.imread(image_path)
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / self.INPUT_WIDTH

        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(self.INPUT_WIDTH, self.INPUT_WIDTH), swapRB=True)
        self.model.setInput(blob)
        outputs = self.model.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25 and maxClassIndex == 0:  # Only signs
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = [round(b * scale) for b in boxes[index]]
            x, y, w, h = box
            roi = original_image[y:y + h, x:x + w].copy()
            # roi_rgb = np.invert(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            # roi_bw = image_preprocessing(roi)/255
            detection = {
                'image': roi,
                'confidence': scores[index]
            }
            detections.append(detection)
        return detections



# class Detector():
#     def __init__(self, detect_threshold=0.5) -> None:
#         self.model_load_time = None
#         self.model = None
#         self.detect_threshold = detect_threshold
#         pass
#
#     def load(self, model_path: str) -> None:
#         start_time = time.time()
#         self.model = tf.saved_model.load(model_path)
#         self.model_load_time = time.time() - start_time
#
#     def detect(self, input_tensor):
#         detections = self.model(input_tensor)
#         num_detections = int(detections["num_detections"])
#         boxes = tf.reshape(detections["detection_boxes"], [
#                            num_detections, 4]).numpy().tolist()
#         scores = tf.reshape(detections["detection_scores"], [
#                             num_detections]).numpy().tolist()
#         classes = tf.reshape(detections["detection_classes"], [
#                              num_detections]).numpy().tolist()
#         return boxes, scores, classes, detections


if __name__ == '__main__':
    detector = Detector()
    detector_model_path = "/home/iv/PycharmProjects/Puzik/Signature/signver-main/models/detector/detection.onnx"
    detector.load(detector_model_path)

    image_path = "/home/iv/PycharmProjects/Puzik/Signature/data/Consistent signature/1_page-0001.jpg"
    detections = detector.detect(image_path)
    for d in detections:
        cv2.imshow("signature", d['image'])
        print("signature confidence = ", d['confidence'])
        cv2.waitKey(0)

