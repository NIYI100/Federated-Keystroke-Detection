from flask import Flask, request
import json

app = Flask(__name__)
@app.route('/', methods=['POST'])
def handle_json():
  content_type = request.headers.get('Content-Type')
  if (content_type == 'application/json'):
    print(request.json)
    return json.dumps(request.json), 200, {'ContentType':'application/json'}
  else:
    return "Content type is not supported."


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080, debug=True)