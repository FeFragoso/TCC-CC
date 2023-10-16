import cv2
import math
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from ultralytics import YOLO

#webcam = model.predict(source='0', show=True)

app = Flask(__name__)
socketio = SocketIO(app)

def ia_cam(input):
    resultado = processa_imagem(input)

    for x in resultado:
        ref, buffer = cv2.imencode('.jpg', x)

        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def processa_imagem(input):
    frame = cv2.VideoCapture(input)
    frame_width = int(frame.get(3))
    frame_height = int(frame.get(4))

    model = YOLO('best.pt')

    classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

    while True:
        success, img = frame.read()

        resultados = model(img, stream=True)

        for x in resultados:
            boxes = x.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                #print(x1, y1, x2, y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

                if (cls == 5):
                    notificacao(classNames[cls])

                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        yield img


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return Response(ia_cam(input=0), mimetype='multipart/x-mixed-replace; boundary=frame')

def notificacao(evento):
    socketio.emit('evento', evento)

if __name__ == "__main__":
    socketio.run(app,allow_unsafe_werkzeug=True, debug=True, host='192.168.100.209', port=80)