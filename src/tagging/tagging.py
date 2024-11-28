import os
import numpy as np
import torch
from torch.nn import functional as TF
import torchaudio
import torchaudio.functional as TAF

from audioset_convnext_inf.pytorch.convnext import ConvNeXt
from audioset_convnext_inf.utils.utilities import read_audioset_label_tags

model_fpath="topel/ConvNeXt-Tiny-AT"

AUDIO_NAME = 'track3.wav'
AUDIO_PATH = os.path.join("./",AUDIO_NAME)

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
  missing = max(audio_target_length - waveform.shape[-1],0)
  waveform = TF.pad(waveform,(0,missing),mode="constant", value=0.0)
elif waveform.shape[-1] > audio_target_length:
  waveform = waveform[:,:audio_target_length]

waveform = waveform.contiguous()
waveform = waveform.to(device)

with torch.no_grad():
  model.eval()
  output = model(waveform)

logits = output["clipwise_logits"]

probs = output["clipwise_output"]

current_dir = os.getcwd()
lb_to_ix, ix_to_lb, id_to_ix, ix_to_id = read_audioset_label_tags(os.path.join(current_dir,"class_labels_indices.csv"))

# Label derivation
threshold = 0.25
sample_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
print(sample_labels)
for l in sample_labels:
    print("%s: %.3f"%(ix_to_lb[l], probs[0,l]))

# Scene level embeddings
with torch.no_grad():
  model.eval()
  output = model.forward_scene_embeddings(waveform)

# Frame level embneddings
with torch.no_grad():
  model.eval()
  output = model.forward_frame_embeddings(waveform)