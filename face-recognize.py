from flask import Flask, render_template, Response
import cv2
import numpy as np
import pickle

app = Flask(__name__, template_folder='template')

class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
    
    def get_frame(self):
        success, frame = self.camera.read()
        if not success:
            return None
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()

    def release(self):
        self.camera.release()

class FaceRecognition(Camera):
    def __init__(self):
        super().__init__()
        self.face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trianner.yml")

        self.labels = {"person_name": 1}
        with open("labels.pickle", "rb") as f:
            og_labels = pickle.load(f)
            self.labels = {v:k for k,v in og_labels.items()}

    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=2.0, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = self.recognizer.predict(roi_gray)
            if conf >= 50 and conf <= 90:
                name = self.labels[id_]
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return frame

class App:
    def __init__(self):
        self.face_recognition = FaceRecognition()

    def generate_frames(self):
        while True:
            frame_bytes = self.face_recognition.get_frame()
            if frame_bytes is None:
                break
            else:
                frame = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame = self.face_recognition.recognize_faces(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def run(self):
        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video')
        def video():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(debug=True)

if __name__ == "__main__":
    app_instance = App()
    app_instance.run()
