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
import requests

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
  avg_data = []
  i = 0
  for clip in clips:
    clip = clip.to(device)
    with torch.no_grad():
      model.eval()
      output = model(clip)
  
      emb_output = model.forward_scene_embeddings(clip)
      emb_output = emb_output[0].tolist()

      # Gettings logits and probabilities
      # logits = output["clipwise_logits"]
      probs = output["clipwise_output"]

      lb_to_ix, ix_to_lb, id_to_ix, ix_to_id = read_audioset_label_tags(os.path.join(current_dir, "class_labels_indices.csv"))
      # Append data for each clip
      clip_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
      clip_data = {
        "clip_position": i,
        "clip_labels": [ix_to_lb[l] for l in clip_labels],
        "clip_probabilities": probs[0].tolist(),
        "clip_embedding": emb_output
      }
      all_clips_data.append(clip_data)
      ++i;
      for label in clip_labels:
        if label not in avg_data:
          avg_data.append({
        "label": ix_to_lb[label],
        "average_probability": probs[0, label].item(),
        "count": 1
          })
        else:
          for data in avg_data:
            if data["label"] == ix_to_lb[label]:
              data["average_probability"] += probs[0, label].item()
              data["count"] += 1

      # Calculate the average probabilities
      for data in avg_data:
        data["average_probability"] /= data["count"]

  # Scene level embeddings
  # with torch.no_grad():
  #   model.eval()


  # # Frame level embeddings
  # with torch.no_grad():
  #   model.eval()
  #   output = model.forward_frame_embeddings(waveform)

  # Initialize OpenSearch client
  client = OpenSearch(
    hosts=[{'host': 'opensearch', 'port': 9200}],
    http_auth=('admin', 'Duck1Teddy#Open'),
    use_ssl=True,
    verify_certs=False,
    # ssl_assert_hostname=False,
    # ssl_show_warn=False,
  )

  # Prepare data to be indexed
  document = {
    "audio_name": audio_name,
    "clip_information": all_clips_data,
    "avg_data": avg_data
  }

  # Index the document
  response = client.index(
    index="audio_labels",
    body=document,
    refresh=True
  )

  print(response)
  
  return ix_to_lb, probs, clip_labels, all_clips_data, avg_data


app = FastAPI()

class AudioRequest(BaseModel):
  audio_name: str

@app.post("/process_audio")
async def process_audio_api(file: UploadFile = File(...)):
  try:
      audio_name = file.filename
      file_location = f"/tmp/{audio_name}"
      with open(file_location, "wb") as f:
          f.write(await file.read())
      tags, probs, sample_labels, all_data, avg_data = process_audio(file_location, audio_name)
      return {"status": "success", "labels": [tags[l] for l in sample_labels], "probabilities": all_data, "average_data": avg_data}
  except Exception as e:
      print(e)
      raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(data: dict):
  try:
    model = data.get("model")
    if not model:
      raise HTTPException(status_code=400, detail="Model data is required")
    return {"status": "success", "message": f"Training started for {model}"}
  except Exception as e:
    print(e)
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/update")
async def update_model(data: dict):
  try:
    return {"status": "success", "message": "Update started"}
  except Exception as e:
    print(e)
    raise HTTPException(status_code=500, detail=str(e))
  

if __name__ == '__main__':
  index_name = "audio_labels"
  url = f"http://localhost:9200/{index_name}"
  headers = {"Content-Type": "application/json"}
  data = {
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 0
    },
    "mappings": {
      "properties": {
        "audio_name": {"type": "keyword"},
        "clip_information": {
         "type": "nested",  # Define clip_information as a nested type
         "properties": {
            "clip_position": { "type": "integer" },
            "clip_labels": { "type": "text" },
            "clip_probabilities": { "type": "float" },
            "clip_embedding": { "type": "dense_vector", "dims": 768 }  # Example dims
          }
        },
        "avg_data": {
          "type": "object",
          "properties": {
            "label": { "type": "text" },                  # Array of text values (supports full-text search)
            "average_probability": { "type": "float" },  # Array of floating-point values
            "count": { "type": "integer" }               # Array of integer values
          }
        },
      }
    }
  }

  response = requests.put(url, headers=headers, json=data)
  print(response.json())
  uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)