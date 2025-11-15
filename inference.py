import fairseq
import torch
import numpy as np
import soundfile as sf

# Since custom user_dir is used during training, need to specifically import the tasks & models
import sys
sys.path.append('/home/student/jingru/projects/se_asr/github_projects/SA-WavLM')
fairseq.tasks.import_tasks('/home/student/jingru/projects/se_asr/github_projects/SA-WavLM/tasks', 'SA-WavLM.tasks')
fairseq.models.import_models('/home/student/jingru/projects/se_asr/github_projects', 'SA-WavLM.models')
import pdb; pdb.set_trace()

# Load checkpoint
ckpt_path = "checkpoints/checkpoint_best.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
model.eval()
# print(model)

# Load audio and speaker embeddings
spk_emb_1 = np.load('spk_emb.npy')
spk_emb_1 = torch.from_numpy(spk_emb_1)
print(spk_emb_1.shape) # 1 x D_spk
audio_path = 'audio.wav'
waveform, _ = sf.read(audio_path)
waveform = torch.from_numpy(waveform).float().unsqueeze(0) # 2. 1 x L

# Get last layer representations
output = model.extract_features(waveform, spk_emb=spk_emb_1)
x = output['x']

# Get all layer representations
layer_results = output['layer_results']
