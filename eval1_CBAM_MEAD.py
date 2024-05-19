from tqdm import tqdm
import torch
import numpy as np
from my_models import generator
from hparams import hparams
import data_loader_MEAD as data_loader
import cv2
import cpbd
from deepface import DeepFace
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
from piq import psnr, ssim, FID, LPIPS
from piq.feature_extractors import InceptionV3

fid_metric = FID()
feature_extractor = InceptionV3() #.cuda()
lpips_metric = LPIPS()
    
def main():
    test_dataloader = data_loader.Create_Dataloader(hparams.test_file, batch_size=1, num_workers=1)
    model = generator.Generator_CBAM()
    if torch.cuda.is_available():
        model = model.cuda()
    load_model(model, None, hparams.pretrained)
    eval(model, test_dataloader)
    
def eval(model, test_dataloader):
    #Validation
    model.eval()
    running_ssim_score = 0.
    running_psnr_score = 0.
    running_fid_score = 0.
    running_lpips_score = 0.
    running_cpbd_score = 0.
    running_csim_score = 0.
    running_lmd_score = 0.
    running_pose_score = 0.
    running_content_score = 0.
    
    prog_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for step, (x, indiv_mels, mel, gt) in prog_bar:
        #x          : [2, 6, 5, 128, 128]
        #indiv_mels : [2, 5, 1, 80, 16]
        #mel        : [2, 1, 80, 16]
        #gt         : [2, 3, 5, 128, 128]
        if torch.cuda.is_available():
            x,indiv_mels, mel, gt = x.cuda(), indiv_mels.cuda(), mel.cuda(), gt.cuda()      
        
        g = model(indiv_mels, x)  # [2, 3, 5, 128, 128]
        
        g = torch.cat([g[:, :, i] for i in range(g.size(2))], dim=0)        # [10, 3, 128, 128]
        gt = torch.cat([gt[:, :, i] for i in range(gt.size(2))], dim=0)
        
        ssim_score = calculate_ssim(g, gt)
        running_ssim_score += ssim_score
        
        psnr_score = calculate_psnr(g, gt)
        running_psnr_score += psnr_score
        
        fid_score = calculate_fid(g, gt)
        running_fid_score += fid_score
        
        lpips_score = calculate_lpips(g, gt)
        running_lpips_score += lpips_score
        
        cpbd_score = calculate_cpbd(g)
        running_cpbd_score += cpbd_score
        
        csim_score = calculate_csim(g, gt, enforce_detection=False)
        running_csim_score += csim_score

        lmd_score, pose_score, content_score = calculate_lmd(g, gt)
        running_lmd_score += lmd_score
        running_pose_score += pose_score
        running_content_score += content_score
        
        prog_bar.set_description('SSIM: {:.3f}; PSNR: {:.3f}; FID: {:.3f}; LPIPS: {:.3f}; CPBD: {:.3f}; CSIM: {:.3f}; LMD: {:.3f}; POSE: {:.3f}; CONTENT: {:.3f} '.format(
            running_ssim_score / (step + 1),
            running_psnr_score / (step + 1),
            running_fid_score / (step + 1),
            running_lpips_score / (step + 1),
            running_cpbd_score / (step + 1),
            running_csim_score / (step + 1),
            running_lmd_score / (step + 1),
            running_pose_score / (step + 1),
            running_content_score / (step + 1),
            ))
        
        #SSIM: 0.920; PSNR: 29.723; FID: 31.532; LPIPS: 0.052; CPBD: 0.309; LMD: 0.006; POSE: 0.003; CONTENT: 0.009
        
        
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


def calculate_ssim(pred, gt):
    ssim_value = ssim(pred, gt, data_range=1., reduction='none')
    return ssim_value.mean().item()

def calculate_fid(pred, gt, FID_batch_size=1024):
    B_mul_T = pred.size(0)
    total_images = torch.cat((gt, pred), 0)
    if len(total_images) > FID_batch_size:
        total_images = torch.split(total_images, FID_batch_size, 0)
    else:
        total_images = [total_images]

    total_feats = []
    for sub_images in total_images:
        if torch.cuda.is_available():
            sub_images = sub_images.cuda()
        feats = fid_metric.compute_feats([
            {'images': sub_images},
        ], feature_extractor=feature_extractor)
        if torch.cuda.is_available():
            feats = feats.detach()
        total_feats.append(feats)
    total_feats = torch.cat(total_feats, 0)
    gt_feat, pd_feat = torch.split(total_feats, (B_mul_T, B_mul_T), 0)

    if torch.cuda.is_available():
        gt_feat = gt_feat.cuda()
        pd_feat = pd_feat.cuda()

    fid = fid_metric.compute_metric(pd_feat, gt_feat)
    return fid.item()

def calculate_psnr(pred, gt):  
    psnr_value = psnr(pred, gt, reduction='none')
    return psnr_value.mean().item()
 
def calculate_lpips(pred, gt):
    lpips_value = lpips_metric(pred, gt)
    return lpips_value.item()

def calculate_cpbd(pred):  
    pred = pred.detach().cpu().numpy()
    total_cpbd = 0
    for batch in range(pred.shape[0]):
        grayscale_image = np.mean(pred[batch], axis=0)
        frame_cpbd = cpbd.compute(grayscale_image)
        total_cpbd += frame_cpbd

    average_cpbd = total_cpbd / (pred.shape[0])
    return average_cpbd

def calculate_csim(pred, gt, model_name='ArcFace', distance_metric='cosine', enforce_detection=True):
    r"""
    model_name: [VGG-Face,OpenFace,Facenet,Facenet512,DeepFace,DeepID,Dlib,ArcFace,SFace]
    distance_metric=[cosine, euclidean, euclidean_l2]
    """
    pred = (pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.) # [10, 128, 128, 3]
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.) # [10, 128, 128, 3]
    total_csim = 0
    for batch in range(pred.shape[0]):
        emb_pred = DeepFace.represent(img_path = pred[batch],model_name = model_name, enforce_detection=enforce_detection)
        emb_pred = emb_pred[0]["embedding"]
        
        emb_gt = DeepFace.represent(img_path = gt[batch],model_name = model_name, enforce_detection=enforce_detection)
        emb_gt = emb_gt[0]["embedding"]
        
        if distance_metric == "cosine":
            csim_score = findCosineDistance(emb_pred, emb_gt)
        elif distance_metric == "euclidean":
            csim_score = findEuclideanDistance(emb_pred, emb_gt)
        elif distance_metric == "euclidean_l2":
            csim_score = findEuclideanDistance(
                l2_normalize(emb_pred), l2_normalize(emb_gt)
            )
        total_csim += csim_score
    average_csim = total_csim / (pred.shape[0] * pred.shape[1])
    return average_csim

def calculate_lmd(pred, gt):
    pred = (pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    total_pose_score = 0.
    total_content_score = 0.
    total_score = 0.
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        for batch in range(pred.shape[0]):
            pred_pose_landmarks, pred_content_landmarks = [], []
            pred_landmarks = []
            gt_pose_landmarks, gt_content_landmarks = [], []
            gt_landmarks = []
            results = face_mesh.process(cv2.cvtColor(pred[batch], cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue  # not detect
            pred_face_landmarks=results.multi_face_landmarks[0]   
            for idx, landmark in enumerate(pred_face_landmarks.landmark):
                if idx in pose_landmark_idx:
                    pred_pose_landmarks.append((landmark.x * 128,landmark.y * 128))
                    pred_landmarks.append((landmark.x * 128,landmark.y * 128))
                if idx in content_landmark_idx:
                    pred_content_landmarks.append((landmark.x * 128,landmark.y * 128))
                    pred_landmarks.append((landmark.x * 128,landmark.y * 128))
            
            results = face_mesh.process(cv2.cvtColor(gt[batch], cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue  # not detect
            gt_face_landmarks=results.multi_face_landmarks[0]
            for idx, landmark in enumerate(gt_face_landmarks.landmark):
                if idx in pose_landmark_idx:
                    gt_pose_landmarks.append((landmark.x * 128,landmark.y * 128))
                    gt_landmarks.append((landmark.x * 128,landmark.y * 128))
                if idx in content_landmark_idx:
                    gt_content_landmarks.append((landmark.x * 128,landmark.y * 128))
                    gt_landmarks.append((landmark.x * 128,landmark.y * 128))

            lmd_pose_score = findLandmarkDistance(pred_pose_landmarks, gt_pose_landmarks)
            lmd_content_score = findLandmarkDistance(pred_content_landmarks, gt_content_landmarks)
            lmd_score = findLandmarkDistance(pred_landmarks, gt_landmarks)
            
            total_pose_score += lmd_pose_score
            total_content_score += lmd_content_score
            total_score += lmd_score

        average_pose = total_pose_score / (pred.shape[0] * pred.shape[1])
        average_content = total_content_score / (pred.shape[0] * pred.shape[1])
        average_score = total_score / (pred.shape[0] * pred.shape[1])
    return average_score, average_pose, average_content

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def findLandmarkDistance(source_representation, test_representation):
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)
        
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance), axis=1)
    euclidean_distance = np.sqrt(euclidean_distance)
    lmd_score = np.mean(euclidean_distance)
    return lmd_score

FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
# (10, 338), (338, 297), (297, 332), (332, 284),(284, 251), (251, 389) (162, 21), (21, 54),(54, 103), (103, 67), (67, 109), (109, 10)

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4), \
                           (4, 45), (45, 220), (220, 115), (115, 48), \
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_FULL = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])
def summarize_landmarks(edge_set):
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks

all_landmark_idx = summarize_landmarks(FACEMESH_FULL)
pose_landmark_idx = \
    summarize_landmarks(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE, \
                                              FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
        [162, 127, 234, 93, 389, 356, 454, 323])
content_landmark_idx = all_landmark_idx - pose_landmark_idx

if __name__ == '__main__':
    hparams.train_file="./data_MEAD/train.txt"
    hparams.test_file="./data_MEAD/test.txt"
    hparams.result_path = './result1_CBAM_MEAD'
    hparams.pretrained = "/media/cxnam/cxnam_folder/HaftFusion/result1_CBAM_MEAD/gen_e195-2024-02-18 05:30:40.839124.pt"

    main()