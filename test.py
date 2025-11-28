#from models import CC_Module
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
import shutil
from tqdm import tqdm
from measure_ssim_psnr import *
# from measure_uiqm import *
from model import D2Mamba # Assuming model is in model_ablations.py
import matplotlib.pyplot as plt

# --- Visualization Utility (Updated) ---
def save_visualizations(base_images, visualizations, feature_map_shape, output_dir="visualizations", file_name_prefix=None):
    """
    Saves visual comparisons of A* paths and saliency maps overlaid on images.

    Args:
        base_images (Tensor): The original batch of input images.
        visualizations (dict): A dictionary from the model containing saliency maps and paths.
        feature_map_shape (tuple): The (H, W) of the feature map to scale paths correctly.
        output_dir (str): Directory to save the output images.
        file_name_prefix (str, optional): A unique prefix (like the original filename) 
                                          to prevent overwriting files.
    """
    if not visualizations:
        print("No visualization data to save (run model in eval mode with 'astar' scan).")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Detach and move tensors to CPU for numpy/matplotlib processing
    base_images = base_images.detach().permute(0, 2, 3, 1).cpu().numpy()
    
    saliency_maps = visualizations['saliency_maps'].cpu().numpy()
    paths_batch = visualizations['paths']
    
    feat_H, feat_W = feature_map_shape
    img_H, img_W = base_images.shape[1], base_images.shape[2]
    
    # Calculate scaling factors to project paths onto the original image size
    scale_h = img_H / feat_H
    scale_w = img_W / feat_W

    for i in range(len(base_images)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 1. Display the original image as the base layer
        ax.imshow(base_images[i])
        
        # 2. Overlay the saliency map as a semi-transparent heatmap
        # Upsample the low-res saliency map to match the original image dimensions
        saliency_resized = np.kron(saliency_maps[i], np.ones((int(scale_h), int(scale_w))))
        ax.imshow(saliency_resized, cmap='jet', alpha=0.5) # 'jet' is a common heatmap colormap
        
        # 3. Overlay the computed A* paths
        paths_for_image = paths_batch[i]
        for path in paths_for_image:
            if not path or len(path) < 2: continue
            # Unzip path coordinates
            r_coords, c_coords = zip(*path)
            # Scale path coordinates from feature map space to original image space
            scaled_c = [c * scale_w + scale_w / 2 for c in c_coords]
            scaled_r = [r * scale_h + scale_h / 2 for r in r_coords]
            ax.plot(scaled_c, scaled_r, marker='o', markersize=3, linestyle='-', linewidth=4, color='red')

        # ax.set_title(f"Image {file_name_prefix or i+1} with Saliency Heatmap and A* Paths")
        ax.axis('off')
        
        plt.tight_layout()
        
        # Construct a unique filename to avoid overwriting
        if file_name_prefix:
            # Sanitize the prefix by removing the file extension
            prefix = os.path.splitext(file_name_prefix)[0]
            save_path = os.path.join(output_dir, f"{prefix}.png")
        else:
            # Fallback for when no prefix is provided
            save_path = os.path.join(output_dir, f"visualization_batch_{i}.png")

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
        # print(f"Saved A* path visualization to {save_path}") # Optional: uncomment for verbose output

def tensor2img(img_tensor):
    """
    Input image tensor shape must be [B C H W]
    the return image numpy array shape is [B H W C]
    """
    res = img_tensor.numpy()
    res = (res + 1.0) / 2.0
    res = np.clip(res, 0.0, 1.0)
    res = res * 255
    res = res.transpose((0,2,3,1))
    return res


CHECKPOINTS_DIR = opt.checkpoints_dir
INP_DIR = opt.testing_dir_inp
CLEAN_DIR = opt.testing_dir_gt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        
#device = 'cpu'
ch = 3

network = D2Mamba(base_dim=24, num_paths=4, mamba_layers=4, scan_strategy_name='astar')
checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,"netG_62.pt"),weights_only=False)
network.load_state_dict(checkpoint['model_state_dict'], strict=False)
network.eval()
network.to(device)

result_dir = './facades/LSUI_FDL_62/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if __name__ =='__main__':
    with torch.no_grad():
        total_files = os.listdir(INP_DIR)
        st = time.time()
        with tqdm(total=len(total_files)) as t:
            total_path_len = 0
            path_len_list=[]
            for m in total_files:
            
                img = cv2.resize(cv2.imread(INP_DIR + str(m)), (256,256), cv2.INTER_CUBIC)
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

                output, path_len, viz_data=network(dataset_torchx)
                output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                output = output[:, :, ::-1]
                cv2.imwrite(os.path.join(result_dir + str(m)), output)

                feature_map_shape = (h // 4, w // 4) 
                save_visualizations(
                    base_images=dataset_torchx, 
                    visualizations=viz_data, 
                    feature_map_shape=feature_map_shape,
                    output_dir="astar_visual_outputs",
                    file_name_prefix=m  # <-- FIX: Pass the unique filename here
                )

                total_path_len = total_path_len + path_len
                path_len_list.append(path_len)

                t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(str(m), h,w, output.shape[0], output.shape[1]))
                t.update(1)
                
        end = time.time()
        print('Total time taken in secs : '+str(end-st))
        print('Per image (avg): '+ str(float((end-st)/len(total_files))))
        print('Path len list: ' + str(path_len_list))
        print('Path len (avg): ' + str(int(total_path_len/len(total_files))))

        
        ### compute SSIM and PSNR
        SSIM_measures, PSNR_measures = SSIMs_PSNRs(CLEAN_DIR, result_dir)
        print("SSIM on {0} samples".format(len(SSIM_measures))+"\n")
        print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures))+"\n")
        print("PSNR on {0} samples".format(len(PSNR_measures))+"\n")
        print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures))+"\n")
        # # inp_uqims = measure_UIQMs(result_dir)
        # # print ("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))
        # # # shutil.rmtree(result_dir)
