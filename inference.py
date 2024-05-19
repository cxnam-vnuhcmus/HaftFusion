import torch
from glob import glob
import os
import numpy as np
import random
from hparams import hparams
import cv2
from my_models import audio
from my_models import generator
from my_models import discriminator
import subprocess
import shutil

def load_file(data_path):
    all_videos = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: 
                line = line.split()[0]
            all_videos.append(line)
    return all_videos

def get_smoothened_mels(mel_chunks, T):
    for i in range(len(mel_chunks)):
        if i > T-1 and i<len(mel_chunks)-T:
            window = mel_chunks[i-T: i + T]
            mel_chunks[i] = np.mean(window, axis=0)
        else:
            mel_chunks[i] = mel_chunks[i]
    return mel_chunks

def prepare_data(all_videos):
    vid_idx = random.randint(0, len(all_videos))
    vid_name = all_videos[vid_idx]
    img_names = list(glob(os.path.join(hparams.face_root, vid_name, '*.png'))) 
    if len(img_names) == 0:
        print(f"Folder is empty: {os.path.join(hparams.face_root, vid_name, '*.png')}")
      
    # aud_idx = random.randint(0, len(all_videos))
    # while aud_idx == vid_idx:
    #     aud_idx = random.randint(0, len(all_videos))
    aud_idx = vid_idx
    aud_name = all_videos[aud_idx]
    wavpath = os.path.join(hparams.audio_root, aud_name, "audio.wav")
    if not os.path.exists(wavpath):
        print(f"Audio file does not exists: {os.path.join(hparams.audio_root, aud_name, 'audio.wav')}")
        
    wav = audio.load_wav(wavpath, hparams.sample_rate)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    mel_chunks = []
    mel_idx_multiplier = 80. / hparams.fps
    mel_step_size = 16
    filter_window = 2
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    if not (filter_window == None):
        mel_chunks = get_smoothened_mels(mel_chunks,T=filter_window)
    print("Length of mel chunks: {}".format(len(mel_chunks)))
    
    gt_img_names = list(glob(os.path.join(hparams.face_root, aud_name, '*.png'))) 
    if len(gt_img_names) == 0:
        print(f"Folder is empty: {os.path.join(hparams.face_root, aud_name, '*.png')}")
    gt_batch = []
    for gt_fname in gt_img_names:
        img = cv2.imread(gt_fname)
        if img is None:
            return None
        try:
            img = cv2.resize(img, (hparams.img_size, hparams.img_size))
        except Exception as e:
            return None
        gt_batch.append(img) 
        
    img_batch, mel_batch = [], []
    for i, m in enumerate(mel_chunks):
        fname = img_names[i%len(img_names)]
        img = cv2.imread(fname)
        if img is None:
            return None
        try:
            img = cv2.resize(img, (hparams.img_size, hparams.img_size))
        except Exception as e:
            return None
        img_batch.append(img) 
        mel_batch.append(m)
    
    img_batch, mel_batch, gt_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(gt_batch)
    
    img_masked = img_batch.copy()
    img_masked[:, hparams.img_size // 2:] = 0
    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    img_batch = np.transpose(img_batch, (3, 0, 1, 2))
    img_batch = np.expand_dims(img_batch, 0)
        
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
    mel_batch = np.transpose(mel_batch, (0, 3, 1, 2))
    mel_batch = np.expand_dims(mel_batch, 0)
    
    gt_batch = np.expand_dims(gt_batch, 0)
    
    img_batch = torch.FloatTensor(img_batch)
    mel_batch = torch.FloatTensor(mel_batch)
    gt_batch = torch.FloatTensor(gt_batch)
    
    return img_batch, mel_batch, gt_batch, vid_idx, aud_idx

def load_model(model, optimizer=None, save_file='.'):
    if next(model.parameters()).is_cuda and torch.cuda.is_available():
        checkpoint = torch.load(save_file, map_location=f'cuda:{torch.cuda.current_device()}')
    else:
        checkpoint = torch.load(save_file, map_location='cpu')
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch = checkpoint["epoch"]
    print(f"Load pretrained model at Epoch: {epoch}")
    return epoch

def fill_blank(g, gt, inps):
    if g.shape[1] < gt.shape[1]:
        black_img = np.zeros((1,gt.shape[1] - g.shape[1],g.shape[-3],g.shape[-2],g.shape[-1])).astype(np.uint8)
        g = np.concatenate((g, black_img), axis=1)
        inps = np.concatenate((inps, black_img), axis=1)
    elif g.shape[1] > gt.shape[1]:
        black_img = np.zeros((1,g.shape[1] - gt.shape[1],g.shape[-3],g.shape[-2],g.shape[-1])).astype(np.uint8)
        gt = np.concatenate((gt, black_img), axis=1)
    return g, gt, inps
    
def save_sample_images(x, g, gt, img_folder, output_video_path):    
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'),
                                      hparams.fps/5, (hparams.img_size * 3, hparams.img_size))
    
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy()).astype(np.uint8)
    refs, inps = x[..., 3:], x[..., :3]    
    
    g, gt, inps = fill_blank(g, gt, inps)
    
    collage = np.concatenate((inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(img_folder, batch_idx, t), c[t])
            out.write(c[t])
    out.release()
          
def merge_audio(video_path, audio_path, output_path):
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, video_path, output_path)
    subprocess.call(command, shell=True)
        
def main():
    all_videos = load_file(hparams.test_file)
    img_batch, mel_batch, gt_batch, vid_idx, aud_idx = prepare_data(all_videos)
    
    model = generator.Generator_CBAM()
    disc = discriminator.HyperCtrolDiscriminator()
    model.eval()
    disc.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        disc = disc.cuda()
        img_batch = img_batch.cuda()
        mel_batch = mel_batch.cuda()
    
    load_model(model, None, '/home/cxnam/Documents/MyWorkingSpace/Pretrained/CREMAD/HaftFusion_Result1_CBAM/gen_e195-2023-12-09 23:30:48.693524.pt')
    
    with torch.no_grad():
        g = model(mel_batch, img_batch)
    
    img_folder = './inference_folder/img_folder'
    output_video_path = './inference_folder/result.avi'
    
    if os.path.exists(img_folder):
        shutil.rmtree(img_folder)        
    
    save_sample_images(img_batch, g, gt_batch, img_folder, output_video_path)
    
    # aud_name = all_videos[aud_idx]
    # wavpath = os.path.join(hparams.audio_root, aud_name, "audio.wav")
    # merge_audio(output_video_path, wavpath, './inference_folder/result_audio.avi')
    
if __name__ == '__main__':
    main()