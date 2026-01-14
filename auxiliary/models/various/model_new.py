import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# ----------------------------
# Utilities
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + h


class FiLMConditioner(nn.Module):
    """Generates FiLM conditioning (gamma, beta) for decoder layers"""
    def __init__(self, latent_dim, num_layers, hidden_dim):
        super().__init__()
        self.film = nn.ModuleList([
            nn.Linear(latent_dim, 2 * hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, latent):
        gammas, betas = [], []
        for layer in self.film:
            g, b = layer(latent).chunk(2, dim=-1)
            gammas.append(g)
            betas.append(b)
        return gammas, betas


# ----------------------------
# Encoders
# ----------------------------
class ResNetEncoderMultiScale(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        base = resnet18(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # 64
        self.layer2 = base.layer2  # 128
        self.layer3 = base.layer3  # 256
        self.layer4 = base.layer4  # 512
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64+128+256+512, out_dim)

    def forward(self, x):
        x = self.stem(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        pooled = torch.cat([self.pool(l).flatten(1) for l in [l1,l2,l3,l4]], dim=1)
        return self.fc(pooled)  # [B, out_dim]


class PointNetEncoder(nn.Module):
    """Light PointNet++-style encoder"""
    def __init__(self, out_dim=1024):
        super().__init__()
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, 256)
        self.mlp4 = nn.Linear(256, out_dim)

    def forward(self, x):  # x: [B, N, 3]
        h = F.relu(self.mlp1(x))
        h = F.relu(self.mlp2(h))
        h = F.relu(self.mlp3(h))
        h = self.mlp4(h)
        h = torch.max(h, dim=1)[0]  # global max pool
        return h


# ----------------------------
# Cross Attention Fusion
# ----------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=1024, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.fc = nn.Linear(dim, dim)

    def forward(self, pc_feat, img_feat):
        # treat both as single tokens
        tokens = torch.stack([pc_feat, img_feat], dim=1)  # [B, 2, D]
        out, _ = self.attn(tokens, tokens, tokens)
        fused = out.mean(dim=1)  # [B, D]
        return self.fc(fused)


# ----------------------------
# Decoder
# ----------------------------
class Decoder(nn.Module):
    def __init__(self, query_dim=3, latent_dim=1024, hidden_dim=256, depth=5, out_dim=1):
        super().__init__()
        self.input_fc = nn.Linear(query_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.film = FiLMConditioner(latent_dim, depth, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, queries, latent):
        # queries: [B, N, query_dim] -> [B*N, query_dim]
        B, N, query_dim = queries.shape
        queries_flat = queries.view(-1, query_dim)
        
        h = self.input_fc(queries_flat)  # [B*N, hidden_dim]
        gammas, betas = self.film(latent)  # each: [B, hidden_dim]
        
        for i, block in enumerate(self.blocks):
            h = block(h)
            # Expand gammas and betas to match flattened queries
            gamma_expanded = gammas[i].unsqueeze(1).expand(-1, N, -1).reshape(-1, gammas[i].size(-1))  # [B*N, hidden_dim]
            beta_expanded = betas[i].unsqueeze(1).expand(-1, N, -1).reshape(-1, betas[i].size(-1))  # [B*N, hidden_dim]
            h = gamma_expanded * h + beta_expanded  # FiLM conditioning
        
        out = self.out(h)  # [B*N, out_dim]
        return out.view(B, N, -1)  # [B, N, out_dim]


# ----------------------------
# Main Network
# ----------------------------
class TreeReconstructionNet(nn.Module):
    def __init__(self, num_points, query_dim=3, bottleneck_size=1024, num_frequencies=8, num_species=2, color_out=False): # query_dim=3, color_out=False):
        super().__init__()
        self.img_enc = ResNetEncoderMultiScale(1024)
        self.pc_enc = PointNetEncoder(1024)
        self.fusion = CrossAttentionFusion(1024)
        self.light_enc = nn.Linear(3, 16)
        self.classifier = nn.Linear(1024, 2)

        self.decoder = Decoder(query_dim=query_dim,
                               latent_dim=1024+16+64,  # fused + light + cat
                               hidden_dim=256, depth=5, out_dim=1)

        self.category_embed = nn.Embedding(2, 64)
        self.color_out = color_out
        if color_out:
            self.color_head = Decoder(query_dim=query_dim,
                                      latent_dim=1024+16+64,
                                      hidden_dim=256, depth=3, out_dim=3)

    def forward(self, dsm_pc, orthophoto, query_points, light_dir=None):
        B = orthophoto.size(0)
        img_feat = self.img_enc(orthophoto)
        pc_feat = self.pc_enc(dsm_pc)
        fused = self.fusion(pc_feat, img_feat)

        # Handle case when light_dir is None
        if light_dir is None:
            light_dir = torch.zeros(B, 3, device=dsm_pc.device)
        light_feat = F.relu(self.light_enc(light_dir))
        
        # Predict category internally
        logits = self.classifier(fused)
        cat_idx = logits.argmax(dim=-1)
        cat_emb = self.category_embed(cat_idx)

        latent = torch.cat([fused, light_feat, cat_emb], dim=-1)

        occ = self.decoder(query_points, latent)
        if self.color_out:
            col = self.color_head(query_points, latent)
            return occ, logits, col
        return occ, logits
