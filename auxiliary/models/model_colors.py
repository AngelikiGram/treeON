import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import pi

# -------------------------------
# Fourier Feature Encoder
# -------------------------------
class FourierEncoder(nn.Module):
    def __init__(self, num_frequencies=25, learnable=False):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_frequencies).float() * pi
        if learnable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs)

    def forward(self, x):
        x = 2.0 * (x - 0.5)  # Normalize to [-1, 1]
        B, N, D = x.shape
        
        # Process in chunks to avoid memory issues with large N
        if N > 16384:  # Only chunk if N is large
            chunk_size = 8192
            encoded_chunks = []
            
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk = x[:, i:end_idx, :]  # [B, chunk_size, D]
                
                x_proj = chunk[..., None] * self.frequencies  # [B, chunk_size, D, F]
                x_proj = x_proj.view(B, end_idx - i, -1)  # [B, chunk_size, D*F]
                encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
                chunk_result = torch.cat([chunk, encoded], dim=-1)  # [B, chunk_size, D + 2*D*F]
                
                encoded_chunks.append(chunk_result)
                del chunk, x_proj, encoded  # Free memory
                
            return torch.cat(encoded_chunks, dim=1)
        else:
            # Original implementation for smaller tensors
            x_proj = x[..., None] * self.frequencies  # [B, N, D, F]
            x_proj = x_proj.view(B, N, -1)  # [B, N, D*F]
            encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return torch.cat([x, encoded], dim=-1)  # [B, N, D + 2*D*F]

# -------------------------------
# Fine Occupancy Refiner
# -------------------------------
class FineOccupancyRefiner(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )

    def forward(self, features):
        return self.refine(features)

# -------------------------------
# Implicit Decoder with Residuals and Color Prediction
# -------------------------------
class ImplicitOccupancyDecoder(nn.Module):
    def __init__(self, bottleneck_size=1024, num_frequencies=25):
        super().__init__()
        self.pe = FourierEncoder(num_frequencies)
        latent_dim = bottleneck_size + 64
        self.pe_dim = 3 + 2 * 3 * num_frequencies  # = 153
        self.input_dim = latent_dim + self.pe_dim  # 2112 + 153 = 2265

        self.fc_in = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.LeakyReLU(0.01), nn.Dropout(0.3))
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.LeakyReLU(0.01))
        self.fc5 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.01))
        self.fc6 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.01))
        self.fc_occ = nn.Linear(64, 1)
        # self.fc_rgb = nn.Linear(64, 3)  # Color prediction

        # COLOR
        # self.fc_rgb = nn.Linear(64 + self.pe_dim, 3) ##

        # COLOR
        self.fc_rgb = nn.Sequential(
            nn.Linear(64 + self.pe_dim, 3),
            nn.Sigmoid()   # ensure outputs in [0,1]
        )

        self.skip_proj = nn.Linear(self.input_dim, 256)

    def forward(self, latent, query_points):
        B, N, _ = query_points.shape
        
        # Process in chunks to reduce memory usage
        chunk_size = min(8192, N)  # Process in smaller chunks
        occupancy_chunks = []
        color_chunks = []
        feature_chunks = []
        
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_points = query_points[:, i:end_idx, :]  # [B, chunk_size, 3]
            chunk_size_actual = end_idx - i
            
            encoded_points = self.pe(chunk_points)  # [B, chunk_size, pe_dim=153]
            latent_expanded = latent.unsqueeze(1).expand(-1, chunk_size_actual, -1)  # [B, chunk_size, 2112]
            x = torch.cat([latent_expanded, encoded_points], dim=-1)  # [B, chunk_size, 2265]
            x = x.view(B * chunk_size_actual, -1) 

            skip = self.skip_proj(x)
            x = F.leaky_relu(self.fc_in(x), negative_slope=0.01)
            x = self.fc2(x)
            x = F.leaky_relu(self.fc3(x) + skip, negative_slope=0.01)
            x = self.fc4(x)
            x = self.fc5(x)
            x = self.fc6(x)

            features_out = x
            occupancy = self.fc_occ(features_out)

            # COLOR
            encoded_flat = encoded_points.view(B * chunk_size_actual, -1)  # [B*chunk_size, pe_dim]
            color_input = torch.cat([features_out, encoded_flat], dim=-1)  # [B*chunk_size, 64 + pe_dim]
            color = self.fc_rgb(color_input)  # [B*chunk_size, 3]

            occupancy_chunks.append(occupancy.view(B, chunk_size_actual, 1))
            color_chunks.append(color.view(B, chunk_size_actual, 3))
            feature_chunks.append(features_out.view(B, chunk_size_actual, -1))
            
            # Clear intermediate variables to free memory
            del x, features_out, encoded_points, latent_expanded, skip, color_input
            torch.cuda.empty_cache()
        
        # Concatenate all chunks
        final_occupancy = torch.cat(occupancy_chunks, dim=1)  # [B, N, 1]
        final_color = torch.cat(color_chunks, dim=1)  # [B, N, 3]
        final_features = torch.cat(feature_chunks, dim=1)  # [B, N, 64]
        
        return final_occupancy, final_color, final_features

# -------------------------------
# ResNet18 Multi-scale Encoder
# -------------------------------
class ResNetEncoderMultiScale(nn.Module):
    def __init__(self, bottleneck_size=1024):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, bottleneck_size)

    def forward(self, x, return_feature_map=False):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        pooled = self.pool(x4).squeeze(-1).squeeze(-1)
        latent = self.fc(pooled)
        if return_feature_map:
            upsampled = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
            return latent, upsampled
        return latent

# -------------------------------
# PointNet Encoder
# -------------------------------
class PointNetEncoder(nn.Module):
    def __init__(self, bottleneck_size=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, bottleneck_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(bottleneck_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, dim=2)[0]
        return x

# -------------------------------
# Full Tree Reconstruction Net
# -------------------------------
class TreeReconstructionNet(nn.Module):
    def __init__(self, num_points, bottleneck_size=1024, num_frequencies=8, num_species=18):
        super().__init__()
        self.pc_encoder = PointNetEncoder(bottleneck_size=bottleneck_size)
        self.img_encoder = ResNetEncoderMultiScale(bottleneck_size=bottleneck_size)
        self.img_fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_size),
            nn.LeakyReLU(0.01),
            nn.Linear(bottleneck_size, bottleneck_size)
        )
        self.decoder = ImplicitOccupancyDecoder(
            bottleneck_size=bottleneck_size * 2,
            num_frequencies=num_frequencies
        )
        self.refiner = FineOccupancyRefiner(input_dim=64)

        # Category embedding (can be used as conditioning vector)
        self.category_predictor = nn.Sequential(
            nn.Linear(bottleneck_size * 2, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),  # learned category embedding
            nn.Tanh()
        )

        # Multi-species classifier (adjustable number of classes)
        self.classifier = nn.Linear(bottleneck_size * 2, num_species)

    def forward(self, dsm_pc, orthophoto, query_points, light_dir=None):
        latent_img = self.img_encoder(orthophoto)
        latent_img = self.img_fc(latent_img)
        latent_img = F.normalize(latent_img, dim=-1)

        latent_pc = self.pc_encoder(dsm_pc)
        latent_pc = F.normalize(latent_pc, dim=-1)

        combined_latent = torch.cat([latent_pc, latent_img], dim=1)
        class_logits = self.classifier(combined_latent)
        category_embed = self.category_predictor(combined_latent)
        latent = torch.cat([combined_latent, category_embed], dim=1)

        coarse_occ, coarse_rgb, coarse_features = self.decoder(latent, query_points)
        # coarse_occ, coarse_features = self.decoder(latent, query_points)
        refined_delta = self.refiner(coarse_features)
        final_occ = coarse_occ + refined_delta

        return final_occ, class_logits, coarse_rgb