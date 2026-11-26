"""RandLA-Net em PyTorch puro (sem operacoes compiladas).

Segue a arquitetura original (Hu et al., CVPR 2020): LocSE + Attentive Pooling
em blocos residuais dilatados, amostragem aleatoria e decoder com interpolacao
pelo vizinho mais proximo. Os indices de vizinhanca sao calculados no
DataLoader (common.build_inputs), como no codigo original.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
git 

def mlp(c_in, c_out, bn=True, act=True):
    layers = [nn.Conv2d(c_in, c_out, 1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out, eps=1e-6, momentum=0.01))
    if act:
        layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def gather(feat, idx):
    """feat (B,C,N,1), idx (B,M,K) -> (B,C,M,K)."""
    B, C, _, _ = feat.shape
    _, M, K = idx.shape
    out = torch.gather(feat.squeeze(-1), 2, idx.reshape(B, 1, M * K).expand(B, C, M * K))
    return out.view(B, C, M, K)


def gather_xyz(xyz, idx):
    """xyz (B,N,3), idx (B,N,K) -> (B,N,K,3)."""
    b = torch.arange(xyz.shape[0], device=xyz.device)[:, None, None]
    return xyz[b, idx]


class LocSE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = mlp(10, d)

    def forward(self, xyz, feat, neigh):
        nxyz = gather_xyz(xyz, neigh)                       # B,N,K,3
        c = xyz.unsqueeze(2).expand_as(nxyz)
        rel = c - nxyz
        dist = rel.norm(dim=-1, keepdim=True)
        enc = torch.cat([dist, rel, c, nxyz], -1).permute(0, 3, 1, 2)  # B,10,N,K
        return torch.cat([self.mlp(enc), gather(feat, neigh)], 1)


class AttPool(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.score = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.mlp = mlp(c_in, c_out)

    def forward(self, x):
        w = F.softmax(self.score(x), dim=-1)
        return self.mlp((x * w).sum(-1, keepdim=True))


class DilatedResBlock(nn.Module):
    def __init__(self, c_in, d_out):
        super().__init__()
        self.mlp1 = mlp(c_in, d_out // 2)
        self.lse1 = LocSE(d_out // 2)
        self.pool1 = AttPool(d_out, d_out // 2)
        self.lse2 = LocSE(d_out // 2)
        self.pool2 = AttPool(d_out, d_out)
        self.mlp2 = mlp(d_out, d_out * 2, act=False)
        self.shortcut = mlp(c_in, d_out * 2, act=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, xyz, feat, neigh):
        f = self.mlp1(feat)
        f = self.pool1(self.lse1(xyz, f, neigh))
        f = self.pool2(self.lse2(xyz, f, neigh))
        return self.act(self.mlp2(f) + self.shortcut(feat))


class RandLANet(nn.Module):
    def __init__(self, d_in=6, num_classes=2, d_out=(16, 64, 128, 256), dropout=0.5):
        super().__init__()
        self.L = len(d_out)
        self.fc0 = mlp(d_in, 8)
        enc_out = [2 * d for d in d_out]
        self.enc = nn.ModuleList()
        c = 8
        for i, d in enumerate(d_out):
            self.enc.append(DilatedResBlock(c, d))
            c = enc_out[i]
        ch = [enc_out[0]] + enc_out              # canais da lista do encoder
        self.dec0 = mlp(ch[-1], ch[-1])
        self.dec = nn.ModuleList()
        cur = ch[-1]
        for j in range(self.L):
            skip = ch[-j - 2]
            self.dec.append(mlp(cur + skip, skip))
            cur = skip
        self.head = nn.Sequential(mlp(cur, 64), mlp(64, 32), nn.Dropout(dropout),
                                  nn.Conv2d(32, num_classes, 1))

    def forward(self, batch):
        xyz, neigh, sub, interp = batch["xyz"], batch["neigh"], batch["sub"], batch["interp"]
        feat = self.fc0(batch["feat"].transpose(1, 2).unsqueeze(-1))   # B,C,N,1
        enc_list = []
        for i in range(self.L):
            f_enc = self.enc[i](xyz[i], feat, neigh[i])
            f_sub = gather(f_enc, sub[i]).max(-1, keepdim=True)[0]      # amostragem aleatoria
            if i == 0:
                enc_list.append(f_enc)
            enc_list.append(f_sub)
            feat = f_sub
        feat = self.dec0(enc_list[-1])
        for j in range(self.L):
            up = gather(feat, interp[self.L - 1 - j])                   # vizinho mais proximo
            feat = self.dec[j](torch.cat([enc_list[-j - 2], up], 1))
        return self.head(feat).squeeze(-1).transpose(1, 2)              # B,N,num_classes