import fairseq
import torch
import numpy as np
import soundfile as sf

# Since custom user_dir is used during training, need to specifically import the tasks & models
import sys
sys.path.append('/home/student/jingru/projects/se_asr/github_projects')
fairseq.tasks.import_tasks('/home/student/jingru/projects/se_asr/github_projects/SA_WavLM/tasks', 'SA_WavLM.tasks')
fairseq.models.import_models('/home/student/jingru/projects/se_asr/github_projects/SA_WavLM/models', 'SA_WavLM.models')

# Load checkpoint
ckpt_path = "/home/student/jingru/projects/se_asr/github_projects/SA_WavLM/checkpoints/checkpoint_best.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
model.eval()
# print(model)

# Load audio and speaker embeddings
spk_emb = torch.rand(1, 192)
waveform = torch.rand(1, 16000)
waveform = waveform.float() # 2. 1 x L

# Get last layer representations
output = model(waveform, spk_emb=spk_emb, features_only=True)
print(output.shape)
