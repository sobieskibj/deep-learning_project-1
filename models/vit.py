import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange

class ViT(nn.Module):

    def __init__(self,
                 image_size = (32, 32),
                 patch_size = (4, 4),
                 num_classes = 10,
                 dim = 30,
                 encoder_depth = 6,
                 channels = 3,
                 embedding_dropout = 0.,
                 **kwargs):
        super(ViT, self).__init__()

        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.img_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches +1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(embedding_dropout)

        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model = dim,
            nhead = encoder_depth,
            batch_first = True,
            **kwargs
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.img_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.positional_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        x = x[:, 0]
        return self.mlp_head(x)

if __name__ == '__main__':

    image_size = (32, 32)
    patch_size = (4, 4)
    num_classes = 10
    dim = 32
    encoder_depth = 2

    model = ViT(
        image_size,
        patch_size,
        num_classes,
        dim,
        encoder_depth,
        channels = 3,
        embedding_dropout = 0.
        )
    
    src = torch.rand(32, 3, 32, 32)
    out = model(src)