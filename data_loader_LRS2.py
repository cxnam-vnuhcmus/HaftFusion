import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import numpy as np
import random
from hparams_LRS2 import hparams
import cv2
from my_models import audio

syncnet_T = hparams.syncnet_T
syncnet_mel_step_size = hparams.syncnet_mel_step_size

class Custom_Dataset(Dataset):
    def __init__(self, data_path):
        self.all_videos = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: 
                    line = line.split()[0]
                self.all_videos.append(line)

    def get_frame_id(self, frame):
        return int(os.path.basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = os.path.dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = os.path.join(vidname, '{:05d}.jpg'.format(frame_id))
            if not os.path.isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        len_all_videos = len(self.all_videos)
        if len_all_videos % hparams.batch_size == 1:
            return len_all_videos - 1
        else:
            return len_all_videos

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            # print(os.path.join(hparams.face_root, vidname, '*.jpg'))
            img_names = list(glob(os.path.join(hparams.face_root, vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue
            
            window = self.read_window(window_fnames)    #[5,128,128,3]
            if window is None:
                continue
            
            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = os.path.join(hparams.audio_root, vidname) + ".wav"
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue
            
            mel = self.crop_audio_window(orig_mel.copy(), img_name) #[16,80]
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name) #[5,80,16]
            if indiv_mels is None: continue

            window = self.prepare_window(window)        #[3,5,128,128]
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.
            

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)  #[6,5,128,128]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y
        
def Create_Dataloader(data_path, batch_size, num_workers=0):
    dataset = Custom_Dataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
