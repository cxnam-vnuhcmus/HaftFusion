from glob import glob
import os

class HParams:
	def __init__(self, **kwargs):
		self.data = {}

		for key, value in kwargs.items():
			self.data[key] = value

	def __getattr__(self, key):
		if key not in self.data:
			raise AttributeError("'HParams' object has no attribute %s" % key)
		return self.data[key]

	def set_hparam(self, key, value):
		self.data[key] = value


hparams = HParams(
    train_file="./data/train.txt",
    test_file="./data/test.txt",
    audio_root="/home/cxnam/Documents/MyWorkingSpace/CREMAD/IPLAP_audio/VideoFPS25",
    face_root= "/home/cxnam/Documents/MyWorkingSpace/CREMAD/IPLAP_face/VideoFPS25/",
    result_path = "./result",
	pretrained = "",
 	disc_pretrained = "",
    disc_multiscale = False,
    
	num_mels=80,
	rescale=True,
	rescaling_max=0.9,
	use_lws=False,
	n_fft=800,
	hop_size=200,
	win_size=800,
	sample_rate=16000,
	frame_shift_ms=None,
	power = 1.5,
	griffin_lim_iters = 60,
	signal_normalization=True,
	allow_clipping_in_normalization=True,
	symmetric_mels=True,
	max_abs_value=4.,
	preemphasize=True,
	preemphasis=0.97,
	
	# Limits
	min_level_db=-100,
	ref_level_db=20,
	fmin=55,
	fmax=7600,

	# Training hyperparameters
	# img_size=256,
 	# img_size=512,
    img_size=128,
	fps=25,
	# batch_size = 32,
 	# batch_size = 28,
  	# batch_size = 16,
  	batch_size = 12,
	initial_learning_rate=1e-4,
    nepochs = 200,
	disc_initial_learning_rate=5e-4,
 	# num_workers=16,
	num_workers=0,


	eval_interval=1000,
	checkpoint_interval=5,
	# eval_interval=300,
	# checkpoint_interval=300,	
	l1_wt = 10.,
	mem_wt=0.2,
	vv_wt = 0.2,
	av_wt=0.2,
	disc_wt=0.2,
	m_slot = 96,
	min = 0,
	max = 0.7,

	syncnet_wt=0.03, # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence. 
	# for pretraining SyncNet
	# syncnet_batch_size=256,
 	syncnet_batch_size=64,
  	# syncnet_batch_size=32,
  	save_optimizer_state=True,
	syncnet_lr=1e-4,
	# syncnet_lr=1e-3,
	# syncnet_eval_interval=10000,
	# syncnet_checkpoint_interval=10000,
	syncnet_eval_interval=5000,
	syncnet_checkpoint_interval=5000,
 	syncnet_T=5,
	syncnet_mel_step_size=16,
)


def hparams_debug_string():
	values = hparams.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)