from tqdm import tqdm
import torch
from torch import nn, optim
import datetime
import os
import json
import numpy as np
import cv2
from my_models import generator
from my_models import discriminator
from hparams import hparams
import data_loader
from my_models import SyncNetLoss
from my_models import GANLoss, PerceptualLoss
from my_models import pix2pixHD_disc

recon_loss = nn.L1Loss()
get_syncnet_loss = SyncNetLoss()
get_gan_loss = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
get_perceptual_loss = PerceptualLoss(network='vgg19',
                    layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
                    num_scales=2)
if torch.cuda.is_available():
    get_syncnet_loss = get_syncnet_loss.cuda()
    get_gan_loss = get_gan_loss.cuda()
    get_perceptual_loss = get_perceptual_loss.cuda()
    
     
def main():
    train_dataloader = data_loader.Create_Dataloader(hparams.train_file, batch_size=hparams.batch_size, num_workers=hparams.num_workers)
    
    model = generator.Generator()
    disc = pix2pixHD_disc.define_D()

    if torch.cuda.is_available():
        model = model.cuda()
        disc = disc.cuda()
    
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))
        
    train(model, optimizer, disc, disc_optimizer, train_dataloader)

def train(model, optimizer, disc, disc_optimizer, train_dataloader):
    current_epoch = 0
    if hparams.pretrained != '':
        current_epoch = load_model(model, optimizer, hparams.pretrained)
    if hparams.disc_pretrained != '':
        load_model(disc, disc_optimizer, hparams.disc_pretrained)
    
    for epoch in range(current_epoch, hparams.nepochs):
        #Training
        
        running_l1_loss = 0.
        running_perceptual_loss = 0.
        running_gan_loss = 0.
        running_gen_loss = 0.
        running_disc_loss = 0.
        running_fake_loss, running_real_loss = 0., 0.
        
        prog_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            #x          : [2, 6, 5, 128, 128]
            #indiv_mels : [2, 5, 1, 80, 16]
            #mel        : [2, 1, 80, 16]
            #gt         : [2, 3, 5, 128, 128]
            if torch.cuda.is_available():
                x,indiv_mels, mel, gt = x.cuda(), indiv_mels.cuda(), mel.cuda(), gt.cuda()   
            
            model.eval()
            disc.train()
            disc_optimizer.zero_grad()

            g = model(indiv_mels, x)  # [2, 3, 5, 128, 128]
            
            if len(g.size()) > 4:
                face_sequences_g = torch.cat([g[:, :, i] for i in range(g.size(2))], dim=0)
                face_sequences_gt = torch.cat([gt[:, :, i] for i in range(gt.size(2))], dim=0)
                
            pred_fake = disc(face_sequences_g.detach())
            loss_D_fake = get_gan_loss(pred_fake, False)
            pred_real = disc(face_sequences_gt.clone().detach())
            loss_D_real = get_gan_loss(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real).mean() * 0.5        
            loss_D.backward()
            disc_optimizer.step()
            
            
            model.train()
            disc.eval()  
            optimizer.zero_grad()
            
            # if len(g.size()) > 4:
            #     face_sequences_g = torch.cat([g[:, :, i] for i in range(g.size(2))], dim=0)
            #     face_sequences_gt = torch.cat([gt[:, :, i] for i in range(gt.size(2))], dim=0)
            perceptual_gen_loss = get_perceptual_loss(face_sequences_g, face_sequences_gt, use_style_loss=True, weight_style_to_perceptual=250).mean()
            l1_loss = recon_loss(g, gt)            
            gan_loss = get_gan_loss(disc(face_sequences_g), True).mean()          
            loss_G = 0.2 * perceptual_gen_loss + 0.4 * gan_loss + (1 - 0.2 - 0.4)*l1_loss
            loss_G.backward()
            optimizer.step()
            
            
            running_l1_loss += l1_loss.item()
            running_perceptual_loss += perceptual_gen_loss.item()
            running_gan_loss += gan_loss.item()
            running_gen_loss += loss_G.item()
            running_disc_loss += loss_D.item()
            running_fake_loss += loss_D_fake.item()
            running_real_loss += loss_D_real.item()

            # prog_bar.set_description('Epoch: {}, Fake: {}, Real: {}'.format(
            #     epoch,
            #     running_fake_loss / (step + 1),
            #     running_real_loss / (step + 1)))
            prog_bar.set_description('Epoch: {}, L1: {:.3f}, Percep: {:.3f}, Gan: {:.3f}, Gen: {:.3f}, Disc: {:.3f}'.format(
                epoch,
                running_l1_loss / (step + 1),
                running_perceptual_loss / (step + 1),
                running_gan_loss / (step + 1),
                running_gen_loss / (step + 1),
                running_disc_loss / (step + 1)))
            
        if epoch % hparams.checkpoint_interval == 0:
            ct = datetime.datetime.now()
            save_model(model, epoch, optimizer, f'{hparams.result_path}/gen_e{epoch}-{ct}.pt')
            save_model(disc, epoch, disc_optimizer, f'{hparams.result_path}/disc_e{epoch}-{ct}.pt')
            save_sample_images(x, g, gt, f'{hparams.result_path}/img_e{epoch}-{ct}')

def save_model(model, epoch, optimizer=None, save_file='.'):
    dir_name = os.path.dirname(save_file)
    os.makedirs(dir_name, exist_ok=True)
    if optimizer is not None:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, str(save_file))
    else:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict()
        }, str(save_file))
        
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

def save_loss(train_loss, val_loss, save_file='.'):
    my_loss = {"train": train_loss,
               "val": val_loss}
    with open(save_file, 'w') as f:
        json.dump(my_loss, f)

def save_sample_images(x, g, gt, folder):    
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]    
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])
      
#MultiscaleDiscriminator
if __name__ == '__main__':
    # hparams.pretrained = '/root/cxnam/HaftFusion/result2/gen_e145-2023-12-02 10:34:35.986715.pt'
    # hparams.disc_pretrained = '/root/cxnam/HaftFusion/result2/disc_e145-2023-12-02 10:34:35.986715.pt'
    hparams.result_path = './result3'
    hparams.disc_multiscale = False
    main()
