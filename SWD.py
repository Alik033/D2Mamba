import torch
import torch.nn as nn

class SpectralWassersteinLoss(nn.Module):
    def __init__(self, n_projections=64, use_lab=True):
        super().__init__()
        self.n_projections = n_projections
        self.use_lab = use_lab
        self.register_buffer("imagenet_means", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("imagenet_stds",  torch.tensor([0.229, 0.224, 0.225]))

    def rgb_to_lab_tensor(self, img):
        B, C, H, W = img.shape
        img_lin = torch.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
        M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]],
                         device=img.device, dtype=img.dtype)
        img_lin = img_lin.permute(0,2,3,1).reshape(-1,3) @ M.T
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        img_xyz = img_lin / torch.tensor([Xn, Yn, Zn], device=img.device, dtype=img.dtype)
        eps = 6/29
        f = torch.where(img_xyz > (eps**3), img_xyz ** (1/3), (img_xyz / (3*eps**2) + 4/29))
        L = 116.0 * f[:,1] - 16.0
        a = 500.0 * (f[:,0] - f[:,1])
        b = 200.0 * (f[:,1] - f[:,2])
        lab = torch.stack([L, a, b], dim=1)
        lab = lab.view(B, H, W, 3).permute(0,3,1,2)
        lab = lab / torch.tensor([100.0, 128.0, 128.0], device=img.device, dtype=img.dtype).view(1,3,1,1)
        return lab

    def forward(self, pred):
        B, C, H, W = pred.shape
        device, dtype = pred.device, pred.dtype

        # Convert pred to Lab if needed
        if self.use_lab:
            pred_colors = self.rgb_to_lab_tensor(pred)
        else:
            pred_colors = pred
        pred_samples = pred_colors.permute(0,2,3,1).reshape(B, -1, C)

        # Ensure ImageNet stats are on the same device/dtype
        means = self.imagenet_means.to(device=device, dtype=dtype)
        stds  = self.imagenet_stds.to(device=device, dtype=dtype)

        # Build synthetic reference samples from ImageNet Gaussian stats
        ref = torch.randn_like(pred_samples) * stds + means
        if self.use_lab:
            ref = self.rgb_to_lab_tensor(ref.permute(0,2,1).view(B,3,1,-1))
            ref = ref.permute(0,2,3,1).reshape(B, -1, C)

        # Random projection directions
        projections = torch.randn(self.n_projections, C, device=device, dtype=dtype)
        projections = projections / (projections.norm(dim=1, keepdim=True) + 1e-8)

        # Compute sliced Wasserstein distance
        swd_batch = []
        for b in range(B):
            proj_pred = pred_samples[b] @ projections.T
            proj_ref  = ref[b] @ projections.T
            proj_pred_sorted, _ = torch.sort(proj_pred, dim=0)
            proj_ref_sorted, _  = torch.sort(proj_ref, dim=0)
            swd_p = torch.mean(torch.abs(proj_pred_sorted - proj_ref_sorted))
            swd_batch.append(swd_p)
        return torch.stack(swd_batch).mean()

