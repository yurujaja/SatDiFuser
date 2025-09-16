import torch
from torch import nn
import torch.nn.functional as F
from archs.aggregation_networks import GlobalWeightedFuser, LocalWeightedFuser, MoEWeightedFuser


class LinearClassifierDecoderMixin:
    def __init__(
            self, 
            projection_dim,
            num_classes,
            *args,  
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.num_classes = num_classes
        
        self.decoder_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # Global average pooling to get (B, C, 1, 1)
            nn.Flatten(),   # Flatten the output to (B, C)
            nn.Linear(projection_dim, num_classes)   # Fully connected layer for classification
        )
    
    def forward(
        self, 
        feats: dict,  
        output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """
        batch_feats: 
        pyramid {ts: [(B, L1, H1, W1), (B, L2, H2, W2), (B, L3, H3, W3), (B, L4, H4, W4)] }
        no pyramid {ts: (B, L, H, W)}
        """
        feats, misc = super().forward(feats)
        b, l, h, w = feats[0].shape
        
        resized_feats = []
        for feat in feats:
            if feat.shape[-2:] != (h, w):
                resized_feats.append(F.interpolate(
                    feat,
                    size=(h,w),
                    mode="bilinear",
                    align_corners=False,
                ))
            else:
                resized_feats.append(feat)
        
        added_feat = sum(resized_feats)
        output = self.decoder_head(added_feat)
        
        return output, misc
    

class GWFuserClassifier(LinearClassifierDecoderMixin, GlobalWeightedFuser):
    def __init__(
            self, 
            feature_dims, 
            projection_dim=384,
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timesteps=[],
            num_classes=1000,
        ):
        super().__init__(
            projection_dim=projection_dim,
            num_classes=num_classes, 
            feature_dims=feature_dims, 
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks, 
            save_timesteps=save_timesteps,
        )


class LWFuserClassifier(LinearClassifierDecoderMixin, LocalWeightedFuser):
    def __init__(
            self, 
            feature_dims, 
            projection_dim=384,
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timesteps=[],
            num_classes=1000,
            gating_tempature=1.0,
        ):
        super().__init__(
            projection_dim=projection_dim,
            num_classes=num_classes, 
            feature_dims=feature_dims, 
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks, 
            save_timesteps=save_timesteps,
            gating_tempature=gating_tempature
        )


class MoEFuserClassifier(LinearClassifierDecoderMixin, MoEWeightedFuser):
    def __init__(
            self, 
            feature_dims, 
            projection_dim=384,
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timesteps=[],
            num_classes=1000,
            num_experts=8,
            top_k=2,
            warmup_steps=300,
            gating_tempature=1.0
        ):
        super().__init__(
            projection_dim=projection_dim,
            num_classes=num_classes, 
            feature_dims=feature_dims, 
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks, 
            save_timesteps=save_timesteps,
            num_experts=num_experts,
            top_k=top_k,
            warmup_steps=warmup_steps,
            gating_tempature=gating_tempature
        )
