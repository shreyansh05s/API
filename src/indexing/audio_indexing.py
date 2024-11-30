from towhee import pipe, ops, DataCollection
import torch
from flask import Flask, request, jsonify

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

p = (
    pipe.input("path")
        .map('path', 'frame', ops.audio_decode.ffmpeg())
        .map('frame', 'vecs', ops.audio_embedding.nnfp(device=device))
        .output('path', 'vecs')
)


app = Flask(__name__)

@app.route('/embed_audio', methods=['POST'])
def embed_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = f"/tmp/{file.filename}"
        file.save(file_path)
        result = DataCollection(p(file_path))
        embeddings = result.to_list()[0]['vecs']
        return jsonify({'embeddings': embeddings.tolist()})
def test_embed_audio():
    with app.test_client() as c:
        response = c.post('/embed_audio', data={'file': open('guitar.wav', 'rb')})
        assert response.status_code == 200
        assert 'embeddings' in response.json

if __name__ == '__main__':
    app.run(debug=True)
