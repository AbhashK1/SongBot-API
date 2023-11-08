from flask import Flask, request, jsonify
from flask_cors import CORS
import chat

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://localhost:4200"}})


@app.route('/chat', methods=['POST'])
def chatx():
    data = request.data.decode('utf-8')
    res = chat.chatRes(data)
    return res


if __name__ == '__main__':
    app.run(port=5000, debug=True)
