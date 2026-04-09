import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import einops

''' In this paper, we propose MBS-CD, a novel multi-scale bottleneck 
spatiotemporal change detection method for remote sensing images. 
The core of MBS-CD is MBSNet, which integrates three key components: 
MGBB for multi-scale feature extraction, BSCA for bidirectional temporal 
correspondence, and SDE-MoE for adaptive expert-based fusion.'''


class Frft_2D(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, x):
        x = fft.fft2(x + self.order * x * x, norm='ortho')
        x = torch.abs(x).float()
        return x


class Encoder_FrFT(nn.Module):
    def __init__(self, inchannel, outchannel, order):
        super(Encoder_FrFT, self).__init__()
        self.frft = nn.Sequential(
            Frft_2D(order=order),
            nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.LeakyReLU(),
        )


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def INF(self, B, H, W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

    def forward(self, q, k, v):
        m_batchsize, _, height, width = q.size()
        proj_query = self.query_conv(q)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(k)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(v)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + v


class DGF_module(nn.Module):
    def __init__(self, dims):
        super(DGF_module, self).__init__()
        self.dims = dims
        self.dec_1_1 = nn.Sequential(
            nn.Conv2d(self.dims[1]*3, self.dims[1], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(),
        )
        self.corss_att_1 = CrissCrossAttention(self.dims[1])
        self.dec_1_2 = nn.Sequential(
            nn.Conv2d(self.dims[1], 1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.LeakyReLU(),
        )

        self.dec_2_1 = nn.Sequential(
            nn.Conv2d(self.dims[2] * 3, self.dims[2], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(),
        )
        self.corss_att_2 = CrissCrossAttention(self.dims[2])
        self.dec_2_2 = nn.Sequential(
            nn.Conv2d(self.dims[2], 1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.LeakyReLU(),
        )

        self.dec_3_1 = nn.Sequential(
            nn.Conv2d(self.dims[3] * 3, self.dims[3], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(),
        )
        self.corss_att_3 = CrissCrossAttention(self.dims[3])
        self.dec_3_2 = nn.Sequential(
            nn.Conv2d(self.dims[3], 1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.LeakyReLU(),
        )

    def forward(self, xA, xB):
        fea_1 = self.dec_1_1(torch.cat([xA[0], xB[0], torch.abs(xA[0] - xB[0])], dim=1))
        fea_1 = self.corss_att_1(xA[0], xB[0], fea_1)
        fea_1 = self.corss_att_1(xA[0], xB[0], fea_1)
        # fea_1 = self.corss_att_1(fea_1, fea_1, fea_1)
        # fea_1 = self.corss_att_1(fea_1, fea_1, fea_1)
        fea_1 = self.dec_1_2(fea_1)

        fea_2 = self.dec_2_1(torch.cat([xA[1], xB[1], torch.abs(xA[1] - xB[1])], dim=1))
        fea_2 = self.corss_att_2(xA[1], xB[1], fea_2)
        fea_2 = self.corss_att_2(xA[1], xB[1], fea_2)
        # fea_2 = self.corss_att_2(fea_2, fea_2, fea_2)
        # fea_2 = self.corss_att_2(fea_2, fea_2, fea_2)
        fea_2 = self.dec_2_2(fea_2)

        fea_3 = self.dec_3_1(torch.cat([xA[2], xB[2], torch.abs(xA[2] - xB[2])], dim=1))
        fea_3 = self.corss_att_3(xA[2], xB[2], fea_3)
        fea_3 = self.corss_att_3(xA[2], xB[2], fea_3)
        # fea_3 = self.corss_att_3(fea_3, fea_3, fea_3)
        # fea_3 = self.corss_att_3(fea_3, fea_3, fea_3)
        fea_3 = self.dec_3_2(fea_3)
        return [fea_1, fea_2, fea_3]


class BottConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            mid_channels, mid_channels, kernel_size, stride, padding, 
            dilation=dilation,  
            groups=mid_channels, 
            bias=False
        )
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class MGBB(nn.Module):
    """Multi-Scale Gated Bottleneck Block (MGB Block)
    
    A lightweight feature encoder with dual-path multi-scale receptive fields,
    channel-wise gating mechanism, and bottleneck convolution for parameter efficiency.
    """
    def __init__(self, in_channels, out_channels, reduction=8):
        super(MGBB, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        mid_channels = max(in_channels // reduction, 16)
    
        self.path_3x3 = nn.Sequential(
            BottConv(in_channels, out_channels, mid_channels, 
                    kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.LeakyReLU(),
            BottConv(out_channels, out_channels, mid_channels, 
                    kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.LeakyReLU()
        )
        
        self.path_5x5 = nn.Sequential(
            BottConv(in_channels, out_channels, mid_channels, 
                    kernel_size=3, stride=1, padding=2, dilation=2), 
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.LeakyReLU()
        )
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.LeakyReLU()
        )
        
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            )
        else:
            self.residual_proj = nn.Identity()
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.2)

    def forward(self, x):
        residual = self.residual_proj(x)

        x1 = self.path_3x3(x)  
        x2 = self.path_5x5(x)  
        gate_weight = self.gate(x)

        x_cat = torch.cat([x1, x2], dim=1)  
        x_fused = self.fusion(x_cat)
        x_gated = x_fused * gate_weight

        output = x_gated + self.alpha * residual
        
        return output


class Ada_fuse(nn.Module):
    def __init__(self, inchannel):
        super(Ada_fuse, self).__init__()
        self.inchannel = inchannel
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_att = nn.Sequential(
            nn.Conv2d(inchannel, inchannel//2, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(inchannel//2, inchannel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
        )

    def forward(self, x1, x2, x3):
        att = torch.cat([x1, x2, x3], dim=1)
        out = self.conv_att(att)

        return out


class BSCA(nn.Module):
    """Bidirectional Spatiotemporal Cross-Attention (BSCA Module)
    
    Establishes explicit correspondence between bi-temporal features through
    coupled spatial-temporal attention, directly yielding discriminative difference representations.
    """
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(BSCA, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )
        self.g2 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        self.W1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.W2 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        g_x11 = self.g1(x1).reshape(batch_size, self.inter_channels, -1)
        g_x21 = self.g2(x2).reshape(batch_size, self.inter_channels, -1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = F.softmax(energy_time_1, dim=-1)
        energy_time_2s = F.softmax(energy_time_2, dim=-1)
        energy_space_2s = F.softmax(energy_space_1, dim=-2)
        energy_space_1s = F.softmax(energy_space_2, dim=-2)

        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()

        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        
        xA_enhanced = x1 + self.W1(y1)  # (B, C, H, W)
        xB_enhanced = x2 + self.W2(y2)  # (B, C, H, W)
        
        diff = xA_enhanced - xB_enhanced  # (B, C, H, W)
        
        return diff

        
class SSFE(nn.Module):
    """Scale-Aware Dual-Expert Mixture of Experts Framework (SSFE Strategy)
    
    A soft MoE module for multi-scale difference feature fusion with:
    1. Per-scale normalization and gating preprocessing
    2. Heterogeneous dual experts (GAP-based global + GMP-based local)
    3. Dynamic routing mechanism with learnable temperature
    4. Contrastive channel fusion strategy
    """
    def __init__(self, dims=[128, 256, 512], reduction=8):
        super(SSFE, self).__init__()
        self.dims = dims
        total_ch = sum(dims)  # 896
        squeeze_ch = max(total_ch // reduction, 64)  # 112
        
        # 每个尺度的输入归一化
        self.scale_norms = nn.ModuleList([
            nn.GroupNorm(num_groups=min(32, d), num_channels=d)
            for d in dims
        ])
        
        # 每个尺度独立的门控
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(d, max(d // 4, 16)),
                nn.GELU(),
                nn.Linear(max(d // 4, 16), d),
                nn.Sigmoid()
            ) for d in dims
        ])
        
        # 拼接后的归一化
        self.concat_norm = nn.GroupNorm(num_groups=32, num_channels=total_ch)
        
        # 全局专家 (AvgPool视角)
        self.global_expert = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_ch, squeeze_ch),
            nn.GELU(),
            nn.Linear(squeeze_ch, total_ch),
        )
        
        # 局部专家 (MaxPool视角)
        self.local_expert = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(total_ch, squeeze_ch),
            nn.GELU(),
            nn.Linear(squeeze_ch, total_ch),
        )
        
        # 动态路由器
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_ch, 2),
            nn.Softmax(dim=-1)
        )
        
        # 可学习温度参数
        self.temp = nn.Parameter(torch.ones(1))
        # 残差缩放
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, d0, d1, d2):
        """
        Args:
            d0: (B, 128, H, W) - 尺度1差异特征
            d1: (B, 256, H, W) - 尺度2差异特征
            d2: (B, 512, H, W) - 尺度3差异特征
        Returns:
            out: (B, 896, H, W) - 融合后特征
        """
        # Step 1: 尺度内归一化 + 门控
        d0 = self.scale_norms[0](d0)
        d1 = self.scale_norms[1](d1)
        d2 = self.scale_norms[2](d2)
        
        d0_w = d0 * self.scale_gates[0](d0).unsqueeze(-1).unsqueeze(-1)
        d1_w = d1 * self.scale_gates[1](d1).unsqueeze(-1).unsqueeze(-1)
        d2_w = d2 * self.scale_gates[2](d2).unsqueeze(-1).unsqueeze(-1)
        
        # Step 2: 拼接 + 归一化
        x = torch.cat([d0_w, d1_w, d2_w], dim=1)  # (B, 896, H, W)
        x = self.concat_norm(x)
        
        # Step 3: 双专家通道权重
        temp = self.temp.clamp(0.5, 2.0)
        w_global = torch.sigmoid(self.global_expert(x) / temp)
        w_local = torch.sigmoid(self.local_expert(x) / temp)
        
        # Step 4: 动态路由
        route = self.router(x)
        alpha, beta = route[:, 0:1], route[:, 1:2]
        
        # Step 5: 对比式通道融合
        w_ch = alpha * w_global + beta * w_local * w_global
        w_ch = w_ch.unsqueeze(-1).unsqueeze(-1)
        
        # Step 6: 通道加权 + 残差
        out = x * w_ch + self.gamma * x
        
        return out


class CD_Model_diff(nn.Module):
    """Change Detection Model with Difference-based Multi-scale Fusion
    
    A change detection network combining MGBB encoding, BSCA attention, 
    and SDE-MoE fusion for remote sensing image change detection.
    """
    def __init__(self, inchannel, patch_size, num_classes=2):
        super(CD_Model_diff, self).__init__()
        self.inchannel = inchannel
        self.patch_size = patch_size
        
        self.dims = [inchannel, 128, 256, 512]

        # Multi-Scale Gated Bottleneck Block (MGBB) encoder
        self.enc = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.enc.append(
                MGBB(
                    in_channels=self.dims[i], 
                    out_channels=self.dims[i+1],
                    reduction=8
                )
            )
        
        # Bidirectional Spatiotemporal Cross-Attention (BSCA) modules
        self.spatiotemporal_attn_modules = nn.ModuleList([
            BSCA(in_channels=self.dims[1]),
            BSCA(in_channels=self.dims[2]),
            BSCA(in_channels=self.dims[3]),
        ])
        
        # Scale-Aware Dual-Expert MoE (SDE-MoE) fusion
        # Total channels: 128 + 256 + 512 = 896
        total_channels = self.dims[1] + self.dims[2] + self.dims[3]  # 896
        
        # SDE-MoE for multi-scale difference feature fusion
        self.expert_fusion = SSFE(
            dims=[self.dims[1], self.dims[2], self.dims[3]],  # [128, 256, 512]
            reduction=8
        )
        
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.cls_head = nn.Linear(total_channels, num_classes)  # 896 -> 2
        self.init_weight()
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, xA, xB):
        d = []

        # Multi-scale encoding with MGBB
        for i in range(len(self.dims)-1):
            xA = self.enc[i](xA)
            xB = self.enc[i](xB)
            
            # BSCA: Bidirectional Spatiotemporal Cross-Attention outputs difference features (B, C, H, W)
            combined = self.spatiotemporal_attn_modules[i](xA, xB)
            
            d.append(combined)
            
        # SDE-MoE: Scale-Aware Dual-Expert MoE fusion for multi-scale features
        weighted_fea = self.expert_fusion(d[0], d[1], d[2])  # (B, 896, H, W)
        
        # Global Max Pooling
        pooled_fea = self.pool(weighted_fea)  # (B, 896, 1, 1)
        pooled_fea = pooled_fea.view(pooled_fea.size(0), -1)  # (B, 896)
        out = self.cls_head(pooled_fea)  # (B, 2)
        
        return out


if __name__ == '__main__':
    """Test model with parameter and FLOPs calculation"""
    xA = torch.randn(1, 224, 5, 5).cuda()  # Time 1 image (B, C, H, W)
    xB = torch.randn(1, 224, 5, 5).cuda()  # Time 2 image (B, C, H, W)

    # Initialize CD_Model_diff with MGBB + BSCA + SDE-MoE
    net = CD_Model_diff(inchannel=224, patch_size=5).cuda()
    # net.train()
    # output = net(xA, xB)
    
    # Calculate parameters and FLOPs
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(net, (xA, xB))
    total = sum([param.nelement() for param in net.parameters()])
    print('Params_Num: {:.2f}M'.format(total/1e6))
    print('FLOPs: {:.2f}M'.format(flops.total()/1e6))

    # print(output.size())
