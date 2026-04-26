import torch
from thop import profile

# ================= Import your models =================
from model2.SegViT_RS import SegViTRS, CFG as SegViTRS_CFG
from model2.HRViT_RS import HRViTRS, CFG as HRViTRS_CFG
from model2.DeepLabVp import DeeplabV3Plus
from model2.GeoDual_Net import Res16_DualDecoder_SegModel, SwinConfig
from model2.SwinUnet import SwinUNet
from model2.TransUnet import TransUNet
from models.UNet import UNet
from models.UperNet import UperNet

# Import the real configuration required by TransUNet
from modelingnew import CONFIGS as CONFIGS_ViT_seg

def check_model_stats():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 6
    img_size = 256

    # Build model dictionary
    models = {}

    # 1. Ours (GeoDual-Net) - keep the real training dimension 64
    cfg = SwinConfig()
    cfg.embed_dim = 64
    cfg.window_size = 4
    models['Ours (GeoDual-Net)'] = Res16_DualDecoder_SegModel(num_classes=num_classes, swin_config=cfg)

    # 2. DeepLabV3+ - standard heavyweight model
    models['DeepLabV3+'] = DeeplabV3Plus(num_classes=num_classes, in_channels=3)

    # ================= Core fix 1: UNet =================
    # Force features=[64, 128, 256, 256] to restore your real lightweight version
    models['UNet'] = UNet(num_classes=num_classes, in_channels=3, features=[64, 128, 256, 256])

    # 4. SwinUNet - standard Tiny version (96 dimensions)
    models['SwinUnet'] = SwinUNet(num_classes=num_classes, in_channels=3, img_size=img_size, embed_dim=96)

    # 5. SegViT - standard version
    cfg_seg = SegViTRS_CFG.copy()
    cfg_seg["in_channels"] = 3
    cfg_seg["num_classes"] = num_classes
    models['SegViT'] = SegViTRS(cfg_seg, img_size=img_size)

    # 6. HRViT - standard ultra-lightweight version
    cfg_hr = HRViTRS_CFG.copy()
    cfg_hr["in_channels"] = 3
    cfg_hr["num_classes"] = num_classes
    models['HRViT'] = HRViTRS(cfg_hr)

    # 7. UperNet - standard heavyweight model
    models['UperNet'] = UperNet(num_classes=num_classes, in_channels=3)

    # ================= Core fix 2: TransUNet =================
    # Force the real config used in training, restore the real small size of 18.65M instead of the 96M skyscraper
    config_trans = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_trans.n_classes = num_classes
    config_trans.hidden_size = 768
    models['TransUNet'] = TransUNet(num_classes=num_classes, in_channels=3, img_size=img_size, config=config_trans)

    # Prepare dummy input (Batch=1, Channels=3, H=256, W=256)
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    print("\n" + "="*50)
    print(f"{'Model Name':<18} | {'Params (M)':<10} | {'FLOPs (G)':<10}")
    print("-" * 50)

    # Loop to compute each model
    stats_dict = {}
    for name, model in models.items():
        model = model.to(device)
        model.eval() # Set evaluation mode
        try:
            # verbose=False prevents printing intermediate layers, keeping console clean
            macs, params = profile(model, inputs=(dummy_input, ), verbose=False)

            # Convert MACs to FLOPs (multiply-accumulate operations count as 2 floating point ops)
            flops = macs * 2

            # Convert to M and G units
            params_M = params / 1e6
            flops_G = flops / 1e9

            # Print results, add a special marker for ours model
            prefix = "⭐ " if "Ours" in name else "   "
            print(f"{prefix}{name:<15} | {params_M:<10.2f} | {flops_G:<10.2f}")

            stats_dict[name] = {'Params': params_M, 'FLOPs': flops_G}

        except Exception as e:
            print(f"   {name:<15} | Computation failed: {e}")

        # Clear memory promptly to prevent OOM when running 8 models sequentially
        del model
        torch.cuda.empty_cache()

    print("="*50 + "\n")
    return stats_dict

if __name__ == "__main__":
    check_model_stats()