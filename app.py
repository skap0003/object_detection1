import os
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return "http://127.0.0.1:5000/api/object_detection"


@app.route('/api/object_detection', methods=['POST'])
def object_detection():
    if not os.path.exists('outputFolder'):
        os.makedirs('outputFolder')

    if 'image' in request.files:
        file = request.files['image']
        fileName = file.filename
        imagePath = os.path.join(app.root_path, 'outputFolder', fileName)
        file.save(imagePath)
        model, classes, output_layers = load_yolo()
        image, height, width, channels = load_image(imagePath)
        blob, outputs = detect_objects(image, model, output_layers)
        boxes, confs, class_ids = get_box_dimensioins(outputs, height, width)
        output = draw_labels(boxes, confs, class_ids, classes)

    return jsonify(output)


def load_yolo():
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, widht, channels = img.shape
    return img, height, widht, channels


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensioins(outputs, height, widht):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0:
                center_x = int(detect[0] * widht)
                center_y = int(detect[1] * height)
                w = int(detect[2] * widht)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
                
    return boxes, confs, class_ids


def draw_labels(boxes, confs, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    output = {}
    objects = []
    for i in range(len(boxes)):
        obj = {}
        if i in indexes:
            obj["label"] = str(classes[class_ids[i]])
            obj["accuracy"] = round((confs[i]*100), 2)
            objects.append(obj)
    output["objects"] = objects

    return output


if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 5001, debug = True)