"""
HGNetV2 Backbone for EdgeCrafter (timm-powered)
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Uses timm (PyTorch Image Models) for the HGNetV2 implementation and pretrained
weights. timm will automatically download weights on first use.

Reference: https://github.com/huggingface/pytorch-image-models
"""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from ..core import register
from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ['HGNetV2']


# ---------------------------------------------------------------------------
# timm HGNetV2 feature channels per stage (strides: [4, 8, 16, 32])
# We select stages [1, 2, 3] ≡ strides [8, 16, 32] for detection/segmentation.
# ---------------------------------------------------------------------------
TIMM_HGNETV2_CHANNELS = {
    'B0': [64, 256, 512, 1024],
    'B1': [64, 256, 512, 1024],
    'B2': [96, 384, 768, 1536],
    'B3': [128, 512, 1024, 2048],
    'B4': [128, 512, 1024, 2048],
    'B5': [128, 512, 1024, 2048],
    'B6': [192, 512, 1024, 2048],
}

VALID_VARIANTS = list(TIMM_HGNETV2_CHANNELS.keys())


@register()
class HGNetV2(nn.Module):
    """HGNetV2 backbone for dense prediction tasks, powered by timm.

    Produces a 3-level feature pyramid at strides [8, 16, 32] by default,
    compatible with HybridEncoder in the EdgeCrafter pipeline.

    On first use, timm automatically downloads pretrained weights from
    HuggingFace Hub to the torch cache directory.

    Args:
        name: variant name ('B0' ~ 'B6')
        pretrained: if True, download pretrained weights from timm (default True)
        weights_path: path to a local .pth checkpoint (overrides pretrained)
        hidden_dim: if set, all output levels are projected to this common dimension.
                    If None, output channels match the backbone's natural channels.
        freeze_stem: freeze stem layers during training
        freeze_at: freeze stages up to this index (0-based, 0=none, 1=stage1, etc.)
        skip_load_backbone: skip loading any pretrained weights (random init)
    """

    def __init__(self,
                 name: str = 'B0',
                 pretrained: bool = True,
                 weights_path: Optional[str] = None,
                 hidden_dim: Optional[int] = None,
                 freeze_stem: bool = False,
                 freeze_at: int = 0,
                 skip_load_backbone: bool = False,
                 **kwargs):
        super().__init__()

        name = name.upper()
        if name not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown HGNetV2 variant '{name}'. "
                f"Available: {VALID_VARIANTS}")

        self.name = name

        # Determine whether to use pretrained weights
        _pretrained = pretrained and not skip_load_backbone

        # If a local checkpoint is provided, we load that after creating the model
        _has_local_ckpt = weights_path is not None and Path(weights_path).exists()

        # Create timm backbone (features_only mode returns multi-scale feature maps)
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for HGNetV2. Install it with: pip install timm")

        self.backbone = timm.create_model(
            f'hgnetv2_{name.lower()}',
            pretrained=_pretrained and not _has_local_ckpt,
            features_only=True,
        )

        # Load local checkpoint if provided
        if _has_local_ckpt:
            self._load_local_checkpoint(weights_path)
        elif not _pretrained and not skip_load_backbone:
            print(
                "=" * 80 + "\n",
                "⚠️  WARNING: HGNetV2 pretrained weights not loaded.\n"
                "Set pretrained=True (default) to auto-download from timm.\n"
                "Training from scratch on HGNetV2 is not recommended.\n",
                "=" * 80,
                sep="")

        # All timm variants output 4 stages at strides [4, 8, 16, 32]
        # We select the last 3 for detection/segmentation → strides [8, 16, 32]
        all_channels = TIMM_HGNETV2_CHANNELS[name]
        self.stage_indices = [1, 2, 3]  # 0-indexed into timm's 4 output levels
        stage_channels = [all_channels[i] for i in self.stage_indices]

        # Channel projection layers
        if hidden_dim is not None:
            self.out_channels = [hidden_dim] * 3
            self.projectors = nn.ModuleList([
                ConvNormLayer_fuse(stage_channels[i], hidden_dim, kernel_size=1, stride=1)
                for i in range(3)
            ])
        else:
            self.out_channels = stage_channels
            self.projectors = nn.ModuleList([
                ConvNormLayer_fuse(stage_channels[i], stage_channels[i], kernel_size=1, stride=1)
                for i in range(3)
            ])

        # Freeze layers if requested
        if freeze_stem and hasattr(self.backbone, 'stem'):
            self._freeze_module(self.backbone.stem)
        for i in range(min(freeze_at, 4)):
            stage_attr = f'stages_{i}'
            if hasattr(self.backbone, stage_attr):
                self._freeze_module(getattr(self.backbone, stage_attr))

    def _load_local_checkpoint(self, weights_path: str):
        """Load weights from a local .pth checkpoint file."""
        path = Path(weights_path)
        state = torch.load(path, map_location='cpu', weights_only=True)

        # Handle common state dict formats
        if 'state_dict' in state:
            state = state['state_dict']
        if 'model' in state:
            state = state['model']

        # Try to strip common prefixes from keys
        new_state = {}
        for k, v in state.items():
            new_k = k
            for prefix in ['backbone.', 'module.', 'model.']:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            new_state[new_k] = v

        missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
        if missing:
            print(f"⚠️  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(
            "=" * 80 + "\n",
            "✅ Local HGNetV2 checkpoint loaded successfully!\n"
            f"📦 Weights file: {path}\n",
            "=" * 80,
            sep="")

    @staticmethod
    def _freeze_module(module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W]

        Returns:
            List of 3 feature maps at strides [8, 16, 32].
        """
        all_feats = self.backbone(x)

        # Select stages [1, 2, 3] → strides [8, 16, 32]
        outs = [all_feats[i] for i in self.stage_indices]

        # Apply optional channel projection
        outs = [proj(feat) for proj, feat in zip(self.projectors, outs)]

        return outs

    def convert_to_deploy(self):
        """Convert model to deployment mode (fuse Conv+BN layers)."""
        self.eval()
        for m in self.children():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
