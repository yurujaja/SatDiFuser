import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from archs.detectron2.resnet import ResNet, BottleneckBlock


class GlobalWeightedFuser(nn.Module):
    """
    Global weighted fusion for the pyramid features, applying learnable module-wise scalars
    """

    def __init__(
            self, 
            feature_dims, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timesteps=[],
        ):
        """
        feature_dims is a dict {w1:[c1, c2, ...], w2:[c1', c2', ...]}
        """
        super().__init__()
             
        self.feature_dims = feature_dims  
        self.scales = sorted(feature_dims.keys(), reverse=True)   # e.g. [64, 32, 16, 8]
        self.num_scales = len(self.scales)     # e.g. 4
        
        self.save_timesteps = save_timesteps

        self.bottleneck_layers = nn.ModuleList()
        
        for scale_idx, scale in enumerate(self.scales):
            for fd_idx, fd in enumerate(self.feature_dims[scale]):
                bottleneck_layer = nn.Sequential(
                    *ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=fd,
                        bottleneck_channels=projection_dim // 4,
                        out_channels=projection_dim,
                        norm="GN",
                        num_norm_groups=num_norm_groups
                    )
                )
                self.bottleneck_layers.append(bottleneck_layer)

        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timesteps))
        self.mixing_weights = nn.Parameter(mixing_weights)
        # self.apply(self.weights_init)   
        

    def weights_init(self, m):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
            

    def get_bottleneck_index(self, stride_id, l, feat_dict):
        """Sum the #layers in all preceding strides + offset by l."""
        return sum(len(feat_dict[s]) for s in self.scales[:stride_id]) + l

    def forward(self, batch_feats):
        """
        Args:
        pyramid batch_feats: 
        {ts: [(B, L1, H1, W1), (B, L2, H2, W2), (B, L3, H3, W3), (B, L4, H4, W4)] }
        Returns:
        scale_feats: [(B, L1, H1, W1), (B, L2, H2, W2), (B, L3, H3, W3), (B, L4, H4, W4)]
        
        H1=64, H2=32, H3=16, H4=8
        """

        mixing_weights = torch.nn.functional.softmax(self.mixing_weights, dim=0)
        sorted_ts = sorted(batch_feats.keys(), reverse=True)
        
        scale_feats = []

        for scale_idx, scale in enumerate(self.scales): 
            scale_feat = None
            for ts_idx, ts in enumerate(sorted_ts):
                t_feats = batch_feats[ts][scale_idx].float()
                start_channel = 0
                for fd_idx, fd in enumerate(self.feature_dims[scale]):
                    end_channel = start_channel + fd
                    feats = t_feats[:, start_channel:end_channel, :, :]
                    
                    bn_layer_idx = self.get_bottleneck_index(scale_idx, fd_idx, self.feature_dims)
                    
                    bottleneck_layer = self.bottleneck_layers[bn_layer_idx]
                    weight_idx = ts_idx * len(self.bottleneck_layers) + bn_layer_idx
                    
                    bottlenecked_feature = bottleneck_layer(feats)
                    bottlenecked_feature = mixing_weights[weight_idx] * bottlenecked_feature
                  
                    if scale_feat is None:
                        scale_feat = bottlenecked_feature
                    else:
                        scale_feat += bottlenecked_feature
                    
                    start_channel = end_channel   
                    # bn_layer_idx += 1
            
            scale_feats.append(scale_feat)  
            
        return scale_feats, mixing_weights


class LocalWeightedFuser(nn.Module):
    """
    Localized weighted fusion for the pyramid features, learns pixel-wise weights 
    """
    
    def __init__(
            self, 
            feature_dims, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timesteps=[],
            gating_tempature=1.0,
        ):
        
        super().__init__()
        
        self.feature_dims = feature_dims    
        self.scales = sorted(self.feature_dims.keys(), reverse=True)
        self.num_scales = len(self.scales)
        self.inscale_cnts = [len(self.feature_dims[w]) for w in self.scales]

        self.save_timesteps = save_timesteps
        self.bottleneck_layers = nn.ModuleList()

        for scale in self.scales:
            for feature_dim in self.feature_dims[scale]:
                bottleneck_layer = nn.Sequential(
                    *ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=feature_dim,
                        bottleneck_channels=projection_dim // 4,
                        out_channels=projection_dim,
                        norm="GN",
                        num_norm_groups=num_norm_groups
                    )
                )
                self.bottleneck_layers.append(bottleneck_layer)
        
        self.gate_modules = nn.ModuleList()
        for i in range(self.num_scales):
            K_s = len(save_timesteps) * self.inscale_cnts[i]
            gating_mod = LocalWeightedPerScale(in_channels=projection_dim, num_modules=K_s, gating_tempature=gating_tempature)
            self.gate_modules.append(gating_mod)

    def forward(self, batch_feats):
        """
        batch_feats is a dict, with key=t, value =list of featurs for each layer, (b, L, w, h)
        each list has length equal to the number of layers
        pyramid {ts: [(B, L1, H1, W1), (B, L2, H2, W2), (B, L3, H3, W3), (B, L4, H4, W4)] }
        H1=64, H2=32, H3=16, H4=8
        """
        
        agg_feats_list = []
        gating_w_list = []
        
        sorted_ts = sorted(batch_feats.keys())[::-1]     # [49, 45, 40, ...]
        
        for scale_id, scale in enumerate(self.scales): 
            agg_feats_TL = []
            for ts in range(len(self.save_timesteps)):
                t_feats = batch_feats[sorted_ts[ts]][scale_id].float()
                start_channel = 0
                agg_feats_L = []
                for l, feature_dim in enumerate(self.feature_dims[scale]):
                    end_channel = start_channel + feature_dim
                    feats = t_feats[:, start_channel:end_channel, :, :]
                    
                    bn_layer_idx = sum(self.inscale_cnts[:scale_id]) + l
                    bottleneck_layer = self.bottleneck_layers[bn_layer_idx]
                    bottlenecked_feature = bottleneck_layer(feats)  # => (B, c', H1, W1)
                  
                    agg_feats_L.append(bottlenecked_feature)      

                agg_feats_L = torch.stack(agg_feats_L, dim=1) # agg_feats_L : (b, L1, c', h1, w1)
                agg_feats_TL.append(agg_feats_L)              
            
            agg_feats_TL = torch.stack(agg_feats_TL, dim=1)  # agg_feats_TL : (b, T, L1, c', h1, w1)
            
            b, t, l, c, h, w = agg_feats_TL.shape
            agg_feats_K = agg_feats_TL.reshape(b, -1, c, h, w)  # agg_feats_K : (b, K1, c', h1, w1)
            agg_feats, gating_w = self.gate_modules[scale_id](agg_feats_K)
            agg_feats_list.append(agg_feats)
            gating_w_list.append(gating_w)
                 
        return agg_feats_list, gating_w_list


class LocalWeightedPerScale(nn.Module):
    def __init__(
        self,
        in_channels,
        num_modules,
        gating_tempature=1.0,
    ):
        """
        :param in_channels: number of channels used by gating conv
        :param num_modules: K = T * L_s
        """
        super().__init__()
        self.gate_conv = nn.Conv2d(in_channels, num_modules, kernel_size=1)
        self.gating_tempature = gating_tempature
        
    def forward(self, feats):
        """
        :param feats: shape [B, K, C, H, W]
        :return fused_feats: [B, C, H, W]
        :return gating_weights: [B, K, H, W]
        """
        B, K, C, H, W = feats.shape
        ref_feat = feats.mean(dim=1)  # => [B, C, H, W]
        
        gating_logits = self.gate_conv(ref_feat)   # => [B, K, H, W]
        gating_weights = F.softmax(gating_logits / self.gating_tempature, dim=1) # => [B, K, H, W]
        gating_weights_expanded = gating_weights.unsqueeze(2)   # => [B, K, 1, H, W]
        fused_feats = (gating_weights_expanded * feats).sum(dim=1)  # => [B, C, H, W]
        
        return fused_feats, gating_weights
    
    
class MoEWeightedFuser(nn.Module):
    def __init__(self, feature_dims, projection_dim=384, num_norm_groups=32, num_res_blocks=1, 
                 save_timesteps=[], num_experts=8, top_k=2, warmup_steps=500, gating_tempature=1.0):
        super().__init__()
        self.feature_dims = feature_dims
        self.scales = sorted(feature_dims.keys(), reverse=True)
        self.num_scales = len(self.scales)
        self.save_timesteps = save_timesteps
         
        self.num_experts = num_experts
        self.top_k = top_k
        self.warmup_steps = warmup_steps

        # Create MOE modules for each scale
        self.scale_moes = nn.ModuleList()
        for scale in self.scales:
            # Calculate input channels for this scale
            in_channels = sum(feature_dims[scale]) 
            scale_moe = MOEWeightedPerScale(
                in_channels=in_channels,
                projection_dim=projection_dim,
                num_experts=num_experts,
                top_k=top_k,
                num_norm_groups=num_norm_groups,
                num_res_blocks=num_res_blocks,
                warmup_steps=warmup_steps,
                gating_tempature=gating_tempature
            )
            self.scale_moes.append(scale_moe)

    def forward(self, batch_feats):
        """
        :param batch_feats: Dict {
            timestep: {
                [tensor(B, C_scale, H_scale, W_scale), ...]
            }
        }
        :return: Dict of {
            'fused': [fused_scale1_tensor, fused_scale2_tensor, ...]
            'gating': [gt_w_scale1_tensor, gt_w_scale2_tensor, ...]
        }
        """
        fused_feats = []
        sorted_ts = sorted(batch_feats.keys(), reverse=True)

        for scale_idx, scale in enumerate(self.scales):
            t_fused_feats, lb_losses = [], 0
            for ts_idx, ts in enumerate(sorted_ts):

                t_feat= batch_feats[ts][scale_idx].float()
                
                fused, lb_loss = self.scale_moes[scale_idx](t_feat)
                t_fused_feats.append(fused)
                lb_losses += lb_loss
            
            fused_feats.append(torch.stack(t_fused_feats, dim=0).mean(dim=0))
        lb_loss_final = lb_losses / (len(self.scales)*len(sorted_ts))
              
        return fused_feats, lb_loss_final      


class MOEWeightedPerScale(nn.Module):
    def __init__(
        self, 
        in_channels, 
        projection_dim, 
        num_experts, 
        top_k, 
        num_norm_groups=32, 
        num_res_blocks=1, 
        warmup_steps=500,
        gating_tempature=1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.projection_dim = projection_dim
        self.warmup_steps = warmup_steps

        # Expert bottleneck layers for this scale
        self.experts = nn.ModuleList([
            nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=in_channels,
                        bottleneck_channels=projection_dim // 4,
                        out_channels=projection_dim,
                        norm="GN",
                        num_norm_groups=num_norm_groups
                )
            ) for _ in range(num_experts)
        ])

        self.gate_network = nn.Sequential(
            nn.Conv2d(in_channels, num_experts, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.temperature = nn.Parameter(torch.tensor(gating_tempature))
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, x):
        """
        :param x: Concatenated features for this scale (B, C_scale, H_scale, W_scale)
        :return: Fused features (B, projection_dim, H_scale, W_scale), gating weights
        """
        B, _, H, W = x.shape
        
        gate_logits = self.gate_network(x)  # (B, num_experts)
        gate_logits = gate_logits / self.temperature
        
        gating_probs = F.softmax(gate_logits, dim=1)
        
        usage = gating_probs.mean(dim=0)
        lb_loss = (usage * torch.log(usage * self.num_experts + 1e-8)).sum()
        lb_loss = lb_loss * (-1.0)
        
        with torch.no_grad():
            self.global_step += 1
        in_warmup = (self.global_step < self.warmup_steps)
        
        if in_warmup:
            output = torch.zeros(B, self.projection_dim, H, W, 
                            device=x.device)
            for e_id, expert in enumerate(self.experts):
                out_e = expert(x)
                w_e = gating_probs[:, e_id].view(B, 1, 1, 1)
                output += w_e * out_e
        
        else:
            topk_scores, topk_indices = torch.topk(gate_logits, k=self.top_k, dim=1)
            gate_weights = F.softmax(topk_scores, dim=1)
            output = torch.zeros(B, self.projection_dim, H, W, 
                                device=x.device)
            for expert_id in range(self.num_experts):
                mask = (topk_indices == expert_id).any(dim=1)
                if not mask.any():
                    continue
                expert_out = self.experts[expert_id](x[mask])
                weights = gate_weights[mask] * (topk_indices[mask] == expert_id).float()
                output[mask] += expert_out * weights.sum(1, keepdim=True)[..., None, None]
            
        return output, lb_loss