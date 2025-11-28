import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import math
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Standalone Mamba Block Implementation (Unchanged) ---
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Projects input x to a higher dimension for main processing (x) and gating (z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        
        # 1D causal convolution to capture local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # Projects convoluted x to get the dynamic SSM parameters (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner)
        
        # Specific projection for the timestep parameter dt
        self.dt_proj = nn.Linear(self.d_state, self.d_inner)

        # Learnable parameters for the SSM state matrix A and feedthrough matrix D
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Projects the final output back to the original model dimension
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x):
        """
        Implements the Mamba forward pass using the optimized parallel scan.
        """
        B, L, d_model = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)

        x_proj_out = self.x_proj(x)
        dt, B_param, C_param = x_proj_out.split([self.d_state, self.d_state, self.d_inner], dim=-1)

        y = self.ssm_scan(x, dt, B_param, C_param)
        
        y = y + x * self.D
        y = y * F.silu(z)
        output = self.out_proj(y)
        return output

    def ssm_scan(self, x, dt, B, C):
        """
        Performs the parallel scan algorithm.
        """
        # Discretize continuous-time SSM parameters
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)
        
        log_dA = torch.log(dA + 1e-8)
        d_log_scan = torch.cumsum(log_dA, dim=1)
        dBx = dB * x.unsqueeze(-1)
        
        d_log_scan_shifted = F.pad(d_log_scan[:, :-1], (0, 0, 0, 0, 1, 0), "constant", 0)
        scan_term = torch.exp(d_log_scan_shifted) * dBx
        h = torch.cumsum(scan_term, dim=1)

        y = (h * C.unsqueeze(-1)).sum(-1)
        return y

# --- 2. A* Pathfinding Module with STABILIZED Cost and GIFH ---
class GeodesicInformationFieldHeuristic(nn.Module):
    """
    Implements the Geodesic Information-Field Heuristic (GIFH).
    This module computes a learnable, physics-inspired heuristic for the A* search.
    """
    def __init__(self, feature_dim, variance_kernel_size=3):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.kernel_size = variance_kernel_size
        self.padding = variance_kernel_size // 2
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=1, padding=self.padding)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel = nn.Parameter(torch.cat([sobel_x, sobel_y], dim=0), requires_grad=False)

    def precompute(self, features, end_node):
        """
        Calculates the heuristic cost grid for a given feature map.
        """
        H, W, C = features.shape
        all_feats = features.reshape(H * W, C)
        
        C_lf = C // 2
        feats_lf = all_feats[:, :C_lf]
        feats_hf = all_feats[:, C_lf:]
        
        features_permuted = features.permute(2, 0, 1)
        grad = F.conv2d(features_permuted.unsqueeze(1), self.sobel.to(features.device), padding=1)
        grad_mag = torch.norm(grad, p=2, dim=1)
        geodesic_grid = torch.mean(grad_mag, dim=0)

        feats_hf_permuted = feats_hf.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)
        mean_sq_hf = self.avg_pool(feats_hf_permuted**2)
        sq_mean_hf = self.avg_pool(feats_hf_permuted)**2
        var_hf = torch.sum(mean_sq_hf - sq_mean_hf, dim=1).squeeze(0)
        info_goal_hf = var_hf[end_node[0], end_node[1]]
        scattering_cost_grid = (info_goal_hf - var_hf)
        
        end_feat_lf = feats_lf.reshape(H, W, -1)[end_node[0], end_node[1]].unsqueeze(0)
        absorption_cost_grid = torch.norm(feats_lf - end_feat_lf, p=2, dim=1).reshape(H, W)

        omega_grid = self.gate_mlp(all_feats).reshape(H, W)

        delta_pos = F.softplus(self.delta)
        gamma_pos = F.softplus(self.gamma)
        beta_pos = F.softplus(self.beta)
        
        heuristic_grid = (delta_pos * geodesic_grid) + \
                         (omega_grid * gamma_pos * scattering_cost_grid) + \
                         ((1 - omega_grid) * beta_pos * absorption_cost_grid)
        
        return F.relu(heuristic_grid)

class AStarPathfinder(nn.Module):
    """
    Implements A* search with a fixed Affinity Cost and a learnable heuristic.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.heuristic_calculator = GeodesicInformationFieldHeuristic(feature_dim)
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.dir_to_idx = {d: i for i, d in enumerate(self.directions)}

    def get_neighbors(self, pos, grid_size_h, grid_size_w):
        x, y = pos
        neighbors = []
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size_h and 0 <= ny < grid_size_w:
                neighbors.append((nx, ny))
        return neighbors

    def precompute_costs(self, features):
        """
        Efficiently precomputes all edge traversal costs for the entire grid.
        """
        H, W, C = features.shape
        cost_grid = torch.full((H, W, 8), float('inf'), device=features.device)
        features_norm = F.normalize(features, p=2, dim=-1)

        for i, (dx, dy) in enumerate(self.directions):
            slice_curr_h = slice(max(0, -dx), min(H, H - dx))
            slice_curr_w = slice(max(0, -dy), min(W, W - dy))
            slice_neigh_h = slice(max(0, dx), min(H, H + dx))
            slice_neigh_w = slice(max(0, dy), min(W, W + dy))
            
            curr_feats = features_norm[slice_curr_h, slice_curr_w]
            neigh_feats = features_norm[slice_neigh_h, slice_neigh_w]
            
            similarity = torch.sum(curr_feats * neigh_feats, dim=-1)
            cost = 1.0 - similarity
            
            cost_grid[slice_curr_h, slice_curr_w, i] = cost
        return cost_grid.cpu().numpy()

    @torch.no_grad()
    def find_path(self, features, start_node, end_node):
        """
        Standard A* search algorithm.
        """
        H, W, _ = features.shape
        heuristic_grid_tensor = self.heuristic_calculator.precompute(features, end_node)
        heuristic_grid = heuristic_grid_tensor.cpu().numpy()
        cost_grid = self.precompute_costs(features)
        
        open_set = []
        heapq.heappush(open_set, (heuristic_grid[start_node], start_node))
        
        came_from = {}
        g_score = { (r, c): float('inf') for r in range(H) for c in range(W) }
        g_score[start_node] = 0
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == end_node:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                return path[::-1]

            for neighbor in self.get_neighbors(current, H, W):
                dx, dy = neighbor[0] - current[0], neighbor[1] - current[1]
                direction_idx = self.dir_to_idx[(dx, dy)]
                cost = cost_grid[current[0], current[1], direction_idx]
                
                tentative_g_score = g_score[current] + cost
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic_grid[neighbor]
                    heapq.heappush(open_set, (f_score, neighbor))
                    
        return None

# --- 3. Core Architectural Components (Unchanged) ---
class ConvAttentionBlock(nn.Module):
    """A standard block combining convolution with spatial attention."""
    def __init__(self, dim, heads=8, attention_downsample_size=8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.AdaptiveAvgPool2d(attention_downsample_size)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.upsample = nn.Upsample(scale_factor=None, mode='bilinear', align_corners=False)

    def forward(self, x):
        x_conv = self.relu(self.norm(self.conv(x)))
        B, C, H, W = x_conv.shape
        identity = x_conv
        x_down = self.downsample(x_conv)
        S = x_down.shape[-1]
        x_flat = x_down.flatten(2).transpose(1, 2)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, S, S)
        self.upsample.size = (H, W)
        attn_out_up = self.upsample(attn_out)
        return identity + attn_out_up

class DualStreamEncoder(nn.Module):
    """
    Encodes the image using two parallel streams: spatial (pixel) and frequency (FFT).
    """
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        self.spatial_enc1 = nn.Sequential(nn.Conv2d(in_channels, base_dim, 3, padding=1), ConvAttentionBlock(base_dim))
        self.spatial_enc2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, 3, padding=1, stride=2), ConvAttentionBlock(base_dim*2))
        self.freq_enc1 = nn.Sequential(nn.Conv2d(2, base_dim, 3, padding=1), nn.ReLU())
        self.freq_enc2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, 3, padding=1, stride=2), nn.ReLU())
        self.fusion1 = nn.Conv2d(base_dim*2, base_dim, 1)
        self.fusion2 = nn.Conv2d(base_dim*4, base_dim*2, 1)
        self.final_conv = nn.Conv2d(base_dim*2, base_dim*4, 3, padding=1, stride=2)
        self.feature_dim = base_dim*4

    def forward(self, x):
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_mag, x_fft_phase = torch.abs(x_fft), torch.angle(x_fft)
        x_fft_input = torch.stack([x_fft_mag.mean(1), x_fft_phase.mean(1)], dim=1)
        s1 = self.spatial_enc1(x)
        f1 = self.freq_enc1(x_fft_input)
        fused1 = self.fusion1(torch.cat([s1, f1], dim=1))
        s2 = self.spatial_enc2(fused1)
        f2 = self.freq_enc2(f1)
        fused2 = self.fusion2(torch.cat([s2, f2], dim=1))
        final_features = self.final_conv(fused2)
        return final_features

class GatedSynthesisDecoder(nn.Module):
    """
    Reconstructs the final image from original and Mamba-corrected features.
    """
    def __init__(self, feature_dim=256, base_dim=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, base_dim*2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(base_dim*2, base_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base_dim, 3, kernel_size=3, padding=1), nn.Sigmoid()
        )
    def forward(self, original_features, mamba_corrections):
        fused_input = torch.cat([original_features, mamba_corrections], dim=1)
        gate_map = self.gate(fused_input)
        fused_features = (1 - gate_map) * original_features + gate_map * mamba_corrections
        return self.decoder(fused_features)

# --- 4. Pluggable Scan Strategy Modules ---
class ScanStrategyBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, features, **kwargs):
        raise NotImplementedError

class AStarScanStrategy(ScanStrategyBase):
    def __init__(self, pathfinder, saliency_head, mamba_corrector, num_paths, **kwargs):
        super().__init__()
        self.pathfinder = pathfinder
        self.saliency_head = saliency_head
        self.mamba_corrector = mamba_corrector
        self.num_paths = num_paths

    def forward(self, features):
        B, C, H_feat, W_feat = features.shape
        batch_mamba_corrections = []
        total_path_len = 0
        
        # Data for visualization
        all_paths_for_batch = []
        all_saliency_for_batch = []

        for i in range(B):
            feat_map = features[i].permute(1, 2, 0)
            
            saliency_map = self.saliency_head(features[i].unsqueeze(0)).squeeze()
            flat_saliency = saliency_map.flatten()
            _, indices = torch.topk(flat_saliency, k=self.num_paths * 2, largest=True)
            node_coords = [((idx // W_feat).item(), (idx % W_feat).item()) for idx in indices]
            
            paths = []
            for j in range(self.num_paths):
                start_node, end_node = node_coords[j], node_coords[self.num_paths + j]
                path = self.pathfinder.find_path(feat_map, start_node, end_node)
                if path:
                    paths.append(path)
            
            all_paths_for_batch.append(paths)
            all_saliency_for_batch.append(saliency_map.detach())

            if not paths: paths.append([(0,0), (H_feat-1, W_feat-1)])
            
            max_len = max(len(p) for p in paths) if paths else 0
            if max_len == 0:
                batch_mamba_corrections.append(torch.zeros_like(features[i]))
                continue

            path_features = torch.zeros(len(paths), max_len, C, device=features.device)
            current_path_len = 0
            for p_idx, path in enumerate(paths):
                current_path_len += len(path)
                for step_idx, (r, c) in enumerate(path):
                    path_features[p_idx, step_idx] = feat_map[r, c]
            
            total_path_len += current_path_len
            
            mamba_out = self.mamba_corrector(path_features)
            
            mamba_correction_map = torch.zeros_like(features[i])
            counts = torch.zeros(1, H_feat, W_feat, device=features.device)
            for p_idx, path in enumerate(paths):
                for step_idx, (r, c) in enumerate(path):
                    mamba_correction_map[:, r, c] += mamba_out[p_idx, step_idx]
                    counts[0, r, c] += 1
            
            mamba_correction_map /= counts.clamp(min=1)
            batch_mamba_corrections.append(mamba_correction_map)

        mamba_corrections = torch.stack(batch_mamba_corrections)
        avg_path_len = total_path_len / B if B > 0 else 0
        
        visualizations = {
            "saliency_maps": torch.stack(all_saliency_for_batch),
            "paths": all_paths_for_batch
        }
        
        return mamba_corrections, avg_path_len, visualizations

class RasterScanStrategy(ScanStrategyBase):
    def __init__(self, mamba_corrector, **kwargs):
        super().__init__()
        self.mamba_corrector = mamba_corrector

    def forward(self, features):
        B, C, H, W = features.shape
        path_len = H * W
        
        flat_features = features.flatten(2).transpose(1, 2)
        mamba_out = self.mamba_corrector(flat_features)
        
        mamba_corrections = mamba_out.transpose(1, 2).reshape(B, C, H, W)
        return mamba_corrections, path_len, None

class BidirectionalScanStrategy(ScanStrategyBase):
    def __init__(self, mamba_corrector, **kwargs):
        super().__init__()
        self.mamba_corrector = mamba_corrector

    def forward(self, features):
        B, C, H, W = features.shape
        path_len = H * W
        
        flat_features = features.flatten(2).transpose(1, 2)
        
        # Forward pass
        mamba_out_fwd = self.mamba_corrector(flat_features)
        
        # Backward pass
        mamba_out_bwd = self.mamba_corrector(torch.flip(flat_features, dims=[1]))
        mamba_out_bwd = torch.flip(mamba_out_bwd, dims=[1])
        
        # Merge and reshape
        mamba_out = (mamba_out_fwd + mamba_out_bwd) / 2
        mamba_corrections = mamba_out.transpose(1, 2).reshape(B, C, H, W)
        return mamba_corrections, path_len, None

class CrossScanStrategy(ScanStrategyBase):
    def __init__(self, mamba_corrector, **kwargs):
        super().__init__()
        self.mamba_corrector = mamba_corrector

    def forward(self, features):
        B, C, H, W = features.shape
        path_len = H * W
        
        flat_features = features.flatten(2).transpose(1, 2)
        
        # Top-left to bottom-right
        out1 = self.mamba_corrector(flat_features)

        # Bottom-right to top-left
        out2 = torch.flip(self.mamba_corrector(torch.flip(flat_features, dims=[1])), dims=[1])
        
        # Top-right to bottom-left (permute rows then scan)
        permuted_rows = features.permute(0, 2, 3, 1).reshape(B, H, W, C)
        permuted_rows = torch.flip(permuted_rows, dims=[1]).reshape(B, H*W, C)
        out3_permuted = self.mamba_corrector(permuted_rows)
        out3 = torch.flip(out3_permuted.reshape(B, H, W, C), dims=[1]).reshape(B, H*W, C)

        # Bottom-left to top-right (permute columns then scan)
        permuted_cols = features.permute(0, 3, 2, 1).reshape(B, W, H, C)
        permuted_cols = torch.flip(permuted_cols, dims=[1]).reshape(B, W*H, C)
        out4_permuted = self.mamba_corrector(permuted_cols)
        out4 = torch.flip(out4_permuted.reshape(B, W, H, C), dims=[1]).permute(0, 2, 1, 3).reshape(B, H*W, C)

        mamba_out = (out1 + out2 + out3 + out4) / 4
        mamba_corrections = mamba_out.transpose(1, 2).reshape(B, C, H, W)
        return mamba_corrections, path_len, None

class D2Mamba(nn.Module):
    """
    The main model integrating all components with a pluggable scan strategy.
    """
    def __init__(self, scan_strategy_name='astar', in_channels=3, base_dim=64, mamba_layers=4, num_paths=8):
        super().__init__()
        self.scan_strategy_name = scan_strategy_name

        self.encoder = DualStreamEncoder(in_channels, base_dim)
        feature_dim = self.encoder.feature_dim
        
        self.pathfinder = AStarPathfinder(feature_dim)
        self.saliency_head = nn.Conv2d(feature_dim, 1, 1)
        self.mamba_corrector = nn.Sequential(*[MambaBlock(d_model=feature_dim) for _ in range(mamba_layers)])
        
        self.decoder = GatedSynthesisDecoder(feature_dim, base_dim)

        strategy_map = {
            'astar': AStarScanStrategy,
            'raster': RasterScanStrategy,
            'bidirectional': BidirectionalScanStrategy,
            'cross_scan': CrossScanStrategy,
        }
        
        if scan_strategy_name not in strategy_map:
            raise ValueError(f"Unknown scan strategy: {scan_strategy_name}")

        self.scan_strategy = strategy_map[scan_strategy_name](
            pathfinder=self.pathfinder,
            saliency_head=self.saliency_head,
            mamba_corrector=self.mamba_corrector,
            num_paths=num_paths
        )

    def forward(self, x):
        features = self.encoder(x)
        
        mamba_corrections, path_len, visualizations = self.scan_strategy(features)
        
        output_image = self.decoder(features, mamba_corrections)

        if self.training:
            return output_image, path_len
        else: # self.eval()
            return output_image, path_len, visualizations


