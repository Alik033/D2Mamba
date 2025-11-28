from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
from options import opt
import math
from measure_ssim_psnr import *
import shutil
from tqdm import tqdm
# from measure_uiqm import *
# from setproctitle import setproctitle

# setproctitle("UIE_alik_test")

CHECKPOINTS_DIR = opt.checkpoints_dir
INP_DIR = opt.testing_dir_inp
CLEAN_DIR = opt.testing_dir_gt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        

ch = 3

atmos_net = D2Mamba(base_dim=24, num_paths=4, mamba_layers=4)

result_dir = './facades/uieb/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def fun():
    with torch.no_grad():
        total_files = os.listdir(INP_DIR)
        st = time.time()
        with tqdm(total=len(total_files)) as t:

            for m in total_files:
            
                img = cv2.resize(cv2.imread(INP_DIR + str(m)), (256,256), cv2.INTER_CUBIC)
                # print(m)
                # img = cv2.imread(INP_DIR + str(m))
                img = img[:, :, ::-1]   
                img = np.float32(img) / 255.0
                h,w,c=img.shape

                train_x = np.zeros((1, ch, h, w)).astype(np.float32)

                train_x[0,0,:,:] = img[:,:,0]
                train_x[0,1,:,:] = img[:,:,1]
                train_x[0,2,:,:] = img[:,:,2]
                dataset_torchx = torch.from_numpy(train_x)
                dataset_torchx=dataset_torchx.to(device)

                output, (_,_) =atmos_net(dataset_torchx)
                output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                output = output[:, :, ::-1]
                cv2.imwrite(os.path.join(result_dir + str(m)), output)

                t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(str(m), h,w, output.shape[0], output.shape[1]))
                t.update(1)
                
        end = time.time()
    #    print('Total time taken in secs : '+str(end-st))
    #    print('Per image (avg): '+ str(float((end-st)/len(total_files))))

        ### compute SSIM and PSNR
        SSIM_measures, PSNR_measures = SSIMs_PSNRs(CLEAN_DIR, result_dir)
        # UIQM_measure = measure_UIQMs(result_dir)

        print("SSIM on {0} samples".format(len(SSIM_measures))+"\n")
        print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures))+"\n")
        print("PSNR on {0} samples".format(len(PSNR_measures))+"\n")
        print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures))+"\n")
        # print("UIQM on {0} samples".format(len(UIQM_measure))+"\n")
        # print("Mean: {0} std: {1}".format(np.mean(UIQM_measure), np.std(UIQM_measure))+"\n")
        return np.mean(SSIM_measures), np.std(SSIM_measures), np.mean(PSNR_measures), np.std(PSNR_measures)#, np.mean(UIQM_measure), np.std(UIQM_measure)
    # shutil.rmtree(result_dir)

if __name__ =='__main__':

    start = 151
    file_path = 'log_uieb_3.txt'

    BEST_PSNR = [0.0, 0.0]
    BEST_SSIM = [0.0, 0.0]
    BEST_PSNR_EPOCH = -1
    BEST_SSIM_EPOCH = -1
    if os.path.exists(file_path):
        # If file exists, delete it
        os.remove(file_path)

    for i in range(start, 163):
        print('EPOCH : ', i)
        print()
        checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,"netG_{}.pt".format(i)),weights_only=False)
        atmos_net.load_state_dict(checkpoint['model_state_dict'])
        atmos_net.eval()
        atmos_net.to(device)

        a, b, c, d = fun()

        if c > BEST_PSNR[0] :
            BEST_PSNR = [c, d]
            BEST_PSNR_EPOCH = i 
        if a > BEST_SSIM[0]:
            BEST_SSIM = [a, b]
            BEST_SSIM_EPOCH = i

        with open(file_path, mode='a') as file:
            file.write('EPOCH {} : SSIM : {} + {}, PSNR : {} + {}'.format(i, a, b, c, d))
            file.write('\n')


    with open(file_path, mode='a') as file:
        file.write('BEST PSNR EPOCH {} & BEST SSIM EPOCH {}: PSNR {} + {}, SSIM : {} + {}'.format(BEST_PSNR_EPOCH, BEST_SSIM_EPOCH, BEST_PSNR[0], BEST_PSNR[1], BEST_SSIM[0], BEST_SSIM[1]))
        file.write('\n')
