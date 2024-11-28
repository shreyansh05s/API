# import torch
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# import librosa

# Load audio
# file_path = "example.wav"
# y, sr = librosa.load(file_path, sr=16000)

# # Load Wav2Vec2 model and processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# # Generate embeddings
# input_values = processor(y, return_tensors="pt", sampling_rate=16000).input_values
# embeddings = model(input_values).last_hidden_state.mean(dim=1).detach().numpy()

from towhee import pipe, ops, DataCollection
import torch


audio = './track04_sour_le_vent.wav'

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

DataCollection(p('track04_sour_le_vent.wav')).show()