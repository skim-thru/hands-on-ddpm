import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimestepEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1

        half_dim = self.embedding_dim // 2
        emb = torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        emb = torch.exp(emb)
        emb = emb.to(timesteps.device)
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super(AttnBlock, self).__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = 1 / (channels ** 0.5)

    def forward(self, x):
        h = self.norm(x)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        B, C, H, W = h.shape
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)

        w = torch.einsum('bci,bcj->bij', q, k) * self.scale
        w = F.softmax(w, dim=-1)

        h = torch.einsum('bij,bcj->bci', w, v).view(B, C, H, W)
        h = self.proj_out(h)

        assert h.shape == x.shape, f"Output shape {h.shape} does not match input shape {x.shape}"

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=128, conv_shortcut=False, dropout=0.0):
        super(ResBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()

        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.temb_act = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()

        self.dropout = dropout
        self.conv_shortcut = conv_shortcut

        if in_channels != out_channels:
            if self.conv_shortcut:
                self.shortcut = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)

        h = h + self.temb_proj(self.temb_act(temb))[:, :, None, None]

        h = self.act2(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)

        x = self.shortcut(h)

        h = h + x
        return h


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=128, conv_shortcut=False, dropout=0.0, apply_attention=False):
        super(UNetBlock, self).__init__()
        self.apply_attention = apply_attention

        self.resblock = ResBlock(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, conv_shortcut=conv_shortcut, dropout=dropout)

        if apply_attention:
            self.attnblock =  AttnBlock(channels=out_channels)
    
    def forward(self, x, temb):
        h = self.resblock(x, temb)
        if self.apply_attention:
            h = self.attnblock(h)

        return h


class UNetModel(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=1,
        attn_resolutions=[16],
        dropout=0.0,
        resamp_with_conv=True,
        num_classes=1,
    ):
        super(UNetModel, self).__init__()

        self.num_classes = num_classes
        self.num_resolutions = len(ch_mult)
        self.ch = ch
        self.resamp_with_conv = resamp_with_conv
        temb_channels = ch * 4

        # Begin
        self.conv_in = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.temb = nn.Sequential(
            TimestepEmbedding(self.ch),
            nn.Linear(self.ch, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )

        # Down path
        self.downblocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        in_channels = ch

        for i_level in range(self.num_resolutions):
            downblocks = nn.ModuleList()
            out_channels = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
                apply_attention = out_channels in attn_resolutions
                downblock = UNetBlock(in_channels, out_channels, temb_channels=temb_channels, dropout=dropout, apply_attention=apply_attention)
                downblocks.append(downblock)
                in_channels = out_channels

            self.downblocks.append(downblocks)

            if i_level != self.num_resolutions - 1:
                downsample_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if resamp_with_conv else nn.AvgPool2d(2, 2)
                self.downsample_layers.append(downsample_layer)

        # Middle path
        self.midblocks = nn.ModuleList([
            UNetBlock(in_channels, in_channels, temb_channels=temb_channels, dropout=dropout, apply_attention=True),
            UNetBlock(in_channels, in_channels, temb_channels=temb_channels, dropout=dropout)
        ])

        # Up path
        self.upblocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        in_channels = ch * ch_mult[-1]
        down_out_channels = self._get_downblock_channels(ch, ch_mult, self.num_resolutions, num_res_blocks)

        for i_level in reversed(range(self.num_resolutions)):
            upblocks = nn.ModuleList()
            out_channels = ch * ch_mult[i_level]

            for _ in range(num_res_blocks + 1):
                apply_attention = out_channels in attn_resolutions
                upblock = UNetBlock(
                    in_channels + down_out_channels.pop(),
                    out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    apply_attention=apply_attention
                )
                upblocks.append(upblock)
                in_channels = out_channels

            self.upblocks.append(upblocks)

            if i_level != 0:
                upsample_layer = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1) if resamp_with_conv else nn.Identity()
                self.upsample_layers.append(upsample_layer)

        assert len(down_out_channels) == 0

        # End
        self.norm_out = nn.GroupNorm(32, in_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1)

    def _get_downblock_channels(self, ch, ch_mult, num_resolutions, num_res_blocks):
        """Helper function to compute the out_channels in downblocks."""
        out_channels_in_downblocks = [ch]

        for i_level in range(num_resolutions):
            out_channels = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                out_channels_in_downblocks.append(out_channels)
            if i_level != num_resolutions - 1:
                out_channels_in_downblocks.append(out_channels)

        return out_channels_in_downblocks

    def forward(self, x, t):
        assert x.dtype == torch.float32

        temb = self.temb(t)

        # Down path
        hs = [self.conv_in(x)]
        h = hs[-1]
        for i_level in range(self.num_resolutions):
            for module in self.downblocks[i_level]:
                h = module(h, temb)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = self.downsample_layers[i_level](h)
                hs.append(h)

        # Middle path
        for module in self.midblocks:
            h = module(h, temb)

        # Up path
        for i_level in range(self.num_resolutions):
            for module in self.upblocks[i_level]:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, temb)

            if i_level != self.num_resolutions - 1:
                h = nn.functional.interpolate(h, scale_factor=2, mode='nearest')
                if self.resamp_with_conv:
                    h = self.upsample_layers[i_level](h)

        # End
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)
        return h
