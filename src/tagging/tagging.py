import os
import numpy as np
import torch
from torch.nn import functional as TF
import torchaudio
import torchaudio.functional as TAF

from audioset_convnext_inf.pytorch.convnext import ConvNeXt
from audioset_convnext_inf.utils.utilities import read_audioset_label_tags
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from opensearchpy import OpenSearch

model_fpath="topel/ConvNeXt-Tiny-AT"

# AUDIO_NAME = 'track3.wav'
def process_audio(file_location, audio_name, model_fpath="topel/ConvNeXt-Tiny-AT", threshold=0.25):
  AUDIO_PATH = file_location
  model = ConvNeXt.from_pretrained(model_fpath, use_auth_token=None, map_location='cpu')

  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  if "cuda" in str(device):
    model = model.to(device)

  sample_rate = 32000
  audio_target_length = 10 * sample_rate

  waveform, sample_rate_ = torchaudio.load(AUDIO_PATH)
  if sample_rate_ != sample_rate:
    waveform = TAF.resample(
      waveform,
      sample_rate_,
      sample_rate,
    )

  if waveform.shape[-1] < audio_target_length:
    missing = max(audio_target_length - waveform.shape[-1], 0)
    waveform = TF.pad(waveform, (0, missing), mode="constant", value=0.0)
  elif waveform.shape[-1] > audio_target_length:
    waveform = waveform[:, :audio_target_length]

  waveform = waveform.contiguous()
  waveform = waveform.to(device)

  with torch.no_grad():
    model.eval()
    output = model(waveform)

  logits = output["clipwise_logits"]

  probs = output["clipwise_output"]

  current_dir = os.getcwd()
  lb_to_ix, ix_to_lb, id_to_ix, ix_to_id = read_audioset_label_tags(os.path.join(current_dir, "class_labels_indices.csv"))

  # Label derivation
  sample_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
  print(sample_labels)
  for l in sample_labels:
    print("%s: %.3f" % (ix_to_lb[l], probs[0, l]))

  # Scene level embeddings
  with torch.no_grad():
    model.eval()
    output = model.forward_scene_embeddings(waveform)

  # Frame level embeddings
  with torch.no_grad():
    model.eval()
    output = model.forward_frame_embeddings(waveform)

  # Initialize OpenSearch client
  client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'admin'),
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
  )

  # Prepare data to be indexed
  document = {
    "audio_name": audio_name,
    "labels": [ix_to_lb[l] for l in sample_labels],
    "probabilities": probs[0].tolist()
  }

  # Index the document
  response = client.index(
    index="audio_labels",
    body=document,
    refresh=True
  )

  print(response)
  
  return ix_to_lb, probs, sample_labels


app = FastAPI()

class AudioRequest(BaseModel):
  file: bytes
  audio_name: str

@app.post("/process_audio")
async def process_audio_api(request: AudioRequest):
  try:
    file_location = f"/tmp/{request.audio_name}"
    tags, probs, sample_labels = process_audio(file_location, request.file,request.audio_name)
    return {"status": "success", "labels": [tags[l] for l in sample_labels], "probabilities": probs[0].tolist()}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8000)