import torch
from einops import repeat


if __name__ == "__main__":
    input_ckpt = "/home/renzhen/userdata/repo/gaiic2024/ckpt/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth"
    save_ckpt = "/home/renzhen/userdata/repo/gaiic2024/ckpt/co_dino_5scale_swin_large_16e_o365tococo-614254c9_patched.pth"

    ckpt = torch.load(input_ckpt, map_location='cpu')

    for ilayer in range(6):
        k = f'query_head.transformer.encoder.layers.{ilayer}.attentions.0.attention_weights'
        ckpt[f'{k}.bias'] = repeat(ckpt[f'{k}.bias'], '(h l p) -> (h m l p)', h=8, l=5, p=4, m=2)
        ckpt[f'{k}.weight'] = repeat(ckpt[f'{k}.weight'], '(h l p) i -> (h m l p) i', h=8, l=5, p=4, m=2)

        k = f'query_head.transformer.encoder.layers.{ilayer}.attentions.0.sampling_offsets'
        ckpt[f'{k}.bias'] = repeat(ckpt[f'{k}.bias'], '(h l p o) -> (h m l p o)', h=8, l=5, p=4, o=2, m=2)
        ckpt[f'{k}.weight'] = repeat(ckpt[f'{k}.weight'], '(h l p o) i -> (h m l p o) i', h=8, l=5, p=4, o=2, m=2)
    torch.save(ckpt, save_ckpt)
