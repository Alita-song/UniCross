"""
3D ViT transformer that inputs 5D (n_batches, n_channels, height, weight, depth)

Based primarily on a video tutorial from Vision Transformer

Official code PyTorch implementation from CDTrans paper:
https://github.com/CDTrans/CDTrans

"""

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc. networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed3D(nn.Module):
    """
    Split image into 3D patches and then embed them.
    img_size: (128, 128, 128)
    proj(b,1,img_size): (b,1,8,8,8)
    n_patches(tokens): 512
    """

    def __init__(self, img_size, patch_size=16, embed_dim=768, patch_embed_fun='conv3d'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)

        if patch_embed_fun == 'conv3d':
            self.proj = nn.Conv3d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, *img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)  --> n_samples: batch size, n_patches: patch numbers
        """
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    """
    Attention mechanism

    Parameters
    -----------
    dim : int (dim per token features):768
    n_heads : int
    qkv_bias : bool
    attn_p : float (Dropout applied to q, k, v)
    proj_p : float (Dropout applied to output tensor)

    Attributes
    ----------
    scale : float
    qkv : nn.Linear
    proj : nn.Linear
    attn_drop, proj_drop : nn.Dropout

    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, n_patches + 1, dim)

        Returns:
        -------
        Shape (n_samples, n_patches + 1, dim)

        """
        B, N, dim = x.shape

        # 1. get Q K V : each with (B, n_heads, N, head_dim)
        qkv = self.qkv(x)  # (B, N, 3 * dim)

        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)  # dim = n_heads * head_dim
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. QK
        dp = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)
        attn = dp.softmax(dim=-1)  # (B, n_heads, N, N)
        attn = self.attn_drop(attn)

        # 3. QKV
        x = (attn @ v).transpose(1, 2).reshape(B, N, dim)  # (B, N, dim)

        # 4. proj
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron

    Parameters
    ----------
    in_features : int
    hidden_features : int
    out_features : int
    p : float

    Attributes
    ---------
    fc1 : nn.Linear
    act : nn.GELU
    fc2 : nn.Linear
    drop : nn.Dropout
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, in_features)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    """
    Transformer block

    Parameters
    ----------
    dim : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p(proj_p), attn_p : float

    Attributes
    ----------
    norm1, norm2 : LayerNorm
    attn : Attention
    mlp : MLP
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop_path_rate=0., p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim
        )

        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, dim)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, dim)
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer

    Parameters
    -----------
    img_size : int
    patch_size : int
    in_chans : int
    n_classes : int
    embed_dim : int
    depth : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p: float pos_drop
    attn_p : float
    """

    def __init__(self,
                 img_size=128,
                 patch_size=16,
                 in_chans=1,
                 n_classes=1000,
                 embed_dim=768,
                 depth=12,
                 n_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_path_rate=0.1,
                 p=0.,
                 attn_p=0.1,
                 patch_embed_fun='conv3d',
                 weight_init='',
                 global_avg_pool=False,
                 use_cls_head=True,
                 ):
        super().__init__()

        if in_chans == 3:
            raise NotImplementedError('3 channels not implemented.')

        if patch_embed_fun in ['conv3d']:
            self.patch_embed = PatchEmbed3D(
                img_size=(img_size, img_size, img_size),
                patch_size=patch_size,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun
            )

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) if not global_avg_pool else None
        embed_len = self.patch_embed.n_patches if global_avg_pool else 1 + self.patch_embed.n_patches  # tokens num
        self.pos_embed = nn.Parameter(
            torch.rand(1, embed_len, embed_dim), requires_grad=True
        )

        self.pos_drop = nn.Dropout(p=p)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=self.dpr[ii],
                    p=p,
                    attn_p=attn_p
                )
                for ii in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

        self.use_cls_head = use_cls_head

    def forward(self, x):
        """
        Input
        -----
        Shape (n_samples, in_chans, img_size, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_classes)

        """
        # batch size
        n_samples = x.shape[0]

        # 1. Patch Embedding
        # input x: (n_samples, in_chans, img_size, img_size, img_size)
        x = self.patch_embed(x)
        # output x: (n_samples, n_patches, embed_dim)

        # 2. add cls token
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(  # cls_token: (1, 1, embed_dim) --> (n_samples, 1, embed_dim)
                n_samples, -1, -1
            )
            x = torch.cat((cls_token, x), dim=1)  # x: (n_samples, 1 + n_patches, embed_dim)

        # 3. position embed
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)

        # 4. dropout
        x = self.pos_drop(x)

        # 5. transformer blocks
        for block in self.blocks:
            x = block(x)

        # 6. final Layer Normalization
        x = self.norm(x)

        # just the CLS token
        cls_token_final = x[:, 0] if self.cls_token is not None else x.mean(dim=1)

        if self.use_cls_head:
            x = self.head(cls_token_final)
        else:
            # x = cls_token_final      # use cls_token
            x = x.mean(dim=1)      # use global average pooling

        return x


def vit_b16(
        img_size=128,
        patch_size=16,
        embed_dim=768,
        in_chans=1,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        p=0.,
        attn_p=0.1,
        global_avg_pool=False,
        patch_embed_fun='conv3d',
        n_classes=2,
        use_cls_head=True,
):
    model = VisionTransformer3D(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_chans=in_chans,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        p=p,
        attn_p=attn_p,
        global_avg_pool=global_avg_pool,
        patch_embed_fun=patch_embed_fun,
        n_classes=n_classes,
        use_cls_head=use_cls_head,
    )
    return model


def vit_b16_backbone(
        img_size=128,
        patch_size=16,
        embed_dim=768,
        in_chans=1,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        p=0.,
        attn_p=0.1,
        global_avg_pool=False,
        patch_embed_fun='conv3d',
        n_classes=2,
        use_cls_head=False,
):
    model = VisionTransformer3D(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_chans=in_chans,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        p=p,
        attn_p=attn_p,
        global_avg_pool=global_avg_pool,
        patch_embed_fun=patch_embed_fun,
        n_classes=n_classes,
        use_cls_head=use_cls_head,
    )
    return model


if __name__ == "__main__":
    model1 = vit_b16_backbone()

    pre_trained_model_path = r"E:\PythonProjects\mlawc\pre_train_model\ViT_B_pretrained_noaug_mae75_BRATS2023_IXI_OASIS3.pth.tar"
    checkpoints = torch.load(pre_trained_model_path, map_location='cpu', weights_only=True)
    print("Loaded pre-trained checkpoint from: %s" % pre_trained_model_path)

    # Extract the state dictionary from checkpoint
    checkpoint_model = checkpoints['net']

    # Load the state dict into your model
    msg = model1.load_state_dict(checkpoint_model, strict=False)

    # Handling possible mismatches
    if msg.missing_keys:
        print("Warning: Missing keys in state dict: ", msg.missing_keys)
    if msg.unexpected_keys:
        print("Warning: Unexpected keys in state dict: ", msg.unexpected_keys)

    img = torch.rand(1, 1, 128, 128, 128).cuda()
    model1.cuda()
    outputs = model1(img)
    print(outputs.size())
