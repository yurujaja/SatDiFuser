import torch
from torch import nn
import torch.nn.functional as F
from archs.aggregation_networks import GlobalWeightedFuser, LocalWeightedFuser, MoEWeightedFuser


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
    """

    def __init__(
        self,
        embed_dim,
        rescales=(4, 2, 1, 0.5),
    ):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        self.ops = nn.ModuleList()

        for i, k in enumerate(self.rescales):
            if k == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                        nn.SyncBatchNorm(embed_dim[i]),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                    )
                )
            elif k == 2:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        )
                    )
                )
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif k == 0.25:
                self.ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise KeyError(f"invalid {k} for feature2pyramid")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []

        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        return tuple(outputs)


class SegUPerNetDecoderMixin:
    def __init__(
        self, 
        feature_dims,
        projection_dim,
        num_norm_groups=32,
        num_res_blocks=1, 
        channels=512,
        num_classes=1000,
        pool_scales=(1, 2, 3, 6),
        rescales=(1,1,1,1),
        align_corners=False,
        save_timesteps=[],
        *args, 
        **kwargs
    ):
        super().__init__(
            feature_dims=feature_dims, 
            projection_dim=projection_dim, 
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks, 
            save_timesteps=save_timesteps,
            *args, 
            **kwargs
        )
        
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        feats_scales = sorted(feature_dims.keys(), reverse=True)     
        num_scales = len(feats_scales)   
        rescales = rescales[:num_scales]
        pool_scales = pool_scales[:num_scales]
        
        self.in_channels = [projection_dim] * num_scales
        
        self.neck = Feature2Pyramid(
            embed_dim=self.in_channels,
            rescales=rescales,
        )

        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    padding=0,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=len(self.in_channels) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)
        
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output
    
    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats
    
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
        
        feats = self.neck(feats)
        feats = self._forward_feature(feats)
        feats = self.dropout(feats)
        output = self.conv_seg(feats)

        assert output_shape is not None

        # interpolate to the target spatial dims
        output = F.interpolate(output, size=output_shape, mode="bilinear")
            
        return output, misc


class GWFuserSegUPerNet(SegUPerNetDecoderMixin, GlobalWeightedFuser):
     def __init__(
            self, 
            feature_dims, 
            projection_dim=384,
            channels=512,
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timesteps=[],
            num_classes=1000,
            pool_scales=(1, 2, 3, 6),
            rescales=(1,1,1,1),
            align_corners=False,
        ):
        super().__init__(
            feature_dims=feature_dims,
            projection_dim=projection_dim,
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks, 
            channels=channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            rescales=rescales,
            align_corners=align_corners,
            save_timesteps=save_timesteps,
        )


class LWFuserSegUPerNet(SegUPerNetDecoderMixin, LocalWeightedFuser):
    def __init__(
            self, 
            feature_dims, 
            projection_dim=384,
            channels=512,
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timesteps=[],
            num_classes=1000,
            pool_scales=(1, 2, 3, 6),
            rescales=(1,1,1,1),
            align_corners=False,
            gating_tempature=1.0
        ):
        super().__init__(
            feature_dims=feature_dims,
            projection_dim=projection_dim,
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks, 
            channels=channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            rescales=rescales,
            align_corners=align_corners,
            save_timesteps=save_timesteps,
            gating_tempature=gating_tempature
        )
  
    
class MoEFuserSegUPerNet(SegUPerNetDecoderMixin, MoEWeightedFuser):
    def __init__(
            self, 
            feature_dims, 
            projection_dim=384,
            channels=512,
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timesteps=[],
            num_classes=1000,
            pool_scales=(1, 2, 3, 6),
            rescales=(1,1,1,1),
            align_corners=False,
            num_experts=8, 
            top_k=2, 
            warmup_steps=500, 
            gating_tempature=1.0
        ):
        super().__init__(
            feature_dims=feature_dims,
            projection_dim=projection_dim,
            num_norm_groups=num_norm_groups,
            num_res_blocks=num_res_blocks, 
            channels=channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            rescales=rescales,
            align_corners=align_corners,
            save_timesteps=save_timesteps,
            num_experts=num_experts,
            top_k=top_k,
            warmup_steps=warmup_steps,
            gating_tempature=gating_tempature
        )