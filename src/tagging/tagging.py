import os
import numpy as np
import torch
from torch.nn import functional as TF
import torchaudio
import torchaudio.functional as TAF

from audioset_convnext_inf.pytorch.convnext import ConvNeXt
from audioset_convnext_inf.utils.utilities import read_audioset_label_tags
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
from opensearchpy import OpenSearch
from uuid import uuid4

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

  # Resampling to 32K 
  waveform, sample_rate_ = torchaudio.load(AUDIO_PATH)
  if sample_rate_ != sample_rate:
    waveform = TAF.resample(
      waveform,
      sample_rate_,
      sample_rate,
    )

  # clipping to the desired length
  # ToDo:   Clip the entire audio file into 10s clips
  if waveform.shape[-1] < audio_target_length:
    missing = max(audio_target_length - waveform.shape[-1], 0)
    waveform = TF.pad(waveform, (0, missing), mode="constant", value=0.0)
  elif waveform.shape[-1] > audio_target_length:
    waveform = waveform[:, :audio_target_length]
    clips = []
    step = sample_rate * 7  # 10 seconds - 3 seconds overlap
    for start in range(0, waveform.shape[-1], step):
      end = start + audio_target_length
      clip = waveform[:, start:end]
      if clip.shape[-1] < audio_target_length:
        missing = max(audio_target_length - clip.shape[-1], 0)
        clip = TF.pad(clip, (0, missing), mode="constant", value=0.0)
      clip = clip.contiguous()
      clips.append(clip)

  # waveform = waveform.contiguous()
  # waveform = waveform.to(device)
  current_dir = os.getcwd()
  all_clips_data = []
  clip_labels = []
  i = 0
  for clip in clips:
    clip = clip.to(device)
    with torch.no_grad():
      model.eval()
      output = model(clip)

      # Gettings logits and probabilities
      logits = output["clipwise_logits"]
      probs = output["clipwise_output"]

      lb_to_ix, ix_to_lb, id_to_ix, ix_to_id = read_audioset_label_tags(os.path.join(current_dir, "class_labels_indices.csv"))
      # Append data for each clip
      clip_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
      clip_data = {
        "clip_position": i,
        "clip_labels": [ix_to_lb[l] for l in clip_labels],
        "clip_probabilities": probs[0].tolist()
      }
      all_clips_data.append(clip_data)
      ++i;
  # Label derivation
  # sample_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
  # print(sample_labels)
  # for l in sample_labels:
  #   print("%s: %.3f" % (ix_to_lb[l], probs[0, l]))

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
    hosts=[{'host': 'opensearch', 'port': 9200}],
    http_auth=('admin', 'Duck1Teddy#Open'),
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
  )

  # Prepare data to be indexed
  document = {
    "audio_name": audio_name,
    "clip_information": all_clips_data
  }

  # Index the document
  response = client.index(
    index="audio_labels",
    body=document,
    refresh=True
  )

  print(response)
  
  return ix_to_lb, probs, clip_labels


app = FastAPI()

class AudioRequest(BaseModel):
  audio_name: str

@app.post("/process_audio")
async def process_audio_api(file: UploadFile = File(...)):
  try:
      audio_name = str(uuid4())
      file_location = f"/tmp/{audio_name}"
      with open(file_location, "wb") as f:
          f.write(await file.read())
      tags, probs, sample_labels = process_audio(file_location, audio_name)
      return {"status": "success", "labels": [tags[l] for l in sample_labels], "probabilities": probs[0].tolist()}
  except Exception as e:
      print(e)
      raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8000)