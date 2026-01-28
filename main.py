from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
from collections import deque, Counter
import random

app = Flask(__name__)

camera = VideoCamera()
emotion_history = deque(maxlen=20)

emotion_quotes = {
    "Happy": ["Happiness looks good on you 😊"],
    "Sad": ["This too shall pass 💙"],
    "Angry": ["Take a deep breath."],
    "Fear": ["Fear is temporary. Courage lasts."],
    "Neutral": ["Stay focused and calm."],
    "Surprise": ["Life is full of surprises!"],
    "Disgust": ["Stay strong and composed."]
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/emotion_data')
def emotion_data():
    emotion = camera.current_emotion
    confidence = camera.current_confidence

    # ✅ ADD EMOTION TO HISTORY
    emotion_history.append(emotion)

    counts = Counter(emotion_history)
    quote = random.choice(emotion_quotes.get(emotion, ["Stay positive!"]))

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "quote": quote,
        "dashboard": dict(counts)
    })


def gen(camera):
    while True:
        frame, _, _ = camera.get_frame()
        if frame is None:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
        )


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset')
def reset():
    emotion_history.clear()
    return ("", 204)


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
