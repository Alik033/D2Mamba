import argparse
import torch
import os

parser = argparse.ArgumentParser()

#parser.add_argument('--data_path', default='/workspace/udit/alik/IEEE_UIE/LOLdataset/our485/')
# UIEB
parser.add_argument('--data_path', default='../../../Dataset/UIEB/UIEB_Train/')
# LSUI
# parser.add_argument('--data_path', default='../../Dataset/UFO120/train_val/')
# EUVP
# parser.add_argument('--data_path', default='/workspace/udit/alik/EUVP/EUVP_Dataset/Paired/')
#SUIM_E
# parser.add_argument('--data_path', default='/workspace/arijit_pg/Alik/Dataset/SUIM/Train/')

parser.add_argument('--checkpoints_dir', default='./ckpt/')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_images', type=int, default=800)

parser.add_argument('--learning_rate_g', type=float, default=2e-04)

parser.add_argument('--end_epoch', type=int, default=201)
parser.add_argument('--img_extension', default='.png')
parser.add_argument('--image_size', type=int ,default=128)

parser.add_argument('--beta1', type=float ,default=0.5)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mse_loss', type=float, default=0.0)
parser.add_argument('--total_mse_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_ssim_loss', type=float, default=0.0)
parser.add_argument('--total_ssim_loss', type=float, default=0.0)

parser.add_argument('--batch_swd_loss', type=float, default=0.0)
parser.add_argument('--total_swd_loss', type=float, default=0.0)

parser.add_argument('--batch_hu_loss', type=float, default=0.0)
parser.add_argument('--total_hu_loss', type=float, default=0.0)

parser.add_argument('--batch_G_loss', type=float, default=0.0)
parser.add_argument('--total_G_loss', type=float, default=0.0)

parser.add_argument('--lambda_mse', type=float, default=1.0)
parser.add_argument('--lambda_vgg', type=float, default=0.02)
parser.add_argument('--lambda_ssim', type=float, default=0.5)
parser.add_argument('--lambda_swd', type=float, default=0.1)


parser.add_argument('--testing_epoch', type=int, default=1)
parser.add_argument('--testing_mode', default="Nat")

# # UIEB
# parser.add_argument('--testing_dir_inp', default="../../../Dataset/UIEB/TEST_HAZY/")
# parser.add_argument('--testing_dir_gt', default="../../../Dataset/UIEB/TEST_CLEAN/")

# LSUI
parser.add_argument('--testing_dir_inp', default="../../../Dataset/LSUI/test/input/")
parser.add_argument('--testing_dir_gt', default="../../../Dataset/LSUI/test/GT/")

# UFO-120
# parser.add_argument('--testing_dir_inp', default="../../Dataset/UFO120/TEST/lrd/")
# parser.add_argument('--testing_dir_gt', default="../../Dataset/UFO120/TEST/hr/")

# U45
# parser.add_argument('--testing_dir_inp', default="../../Dataset/U45/")

# UCCS
# parser.add_argument('--testing_dir_inp', default="../../Dataset/UCCS/")
# C60
# parser.add_argument('--testing_dir_inp', default="../../Dataset/challenging_60/")
# SQUID
# parser.add_argument('--testing_dir_inp', default="../../Dataset/SQUID/Left/")
opt = parser.parse_args()
# print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print(device)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)
