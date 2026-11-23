"""Patch-feature extraction for ring-shaped inspection images (optionally polar-unwrapped).

For every TIFF image:  [crop] -> [polar unwrap] -> JET colour map -> grid patches
-> drop dark/background patches -> frozen ImageNet encoder -> save a bag of patch features.

Output: <output_dir>/<encoder>/<class>/<image>_features.pt            (Cartesian patches)
        <output_dir>/polar/<encoder>/<class>/<image>_features.pt      (with --use_polar)
Each file stores {'features': (N, D) tensor, 'label': int}; one file = one MIL bag.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RingBeamDataset
from model import build_model
from polar import grid_patches, to_rgb, unwrap_ring


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='dataset', help='Root directory with one sub-folder per class.')
    parser.add_argument('--output_dir', type=str, default='features', help='Directory to save the extracted features.')
    parser.add_argument('--encoder', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small'])
    parser.add_argument('--classes', nargs='+', help='Class sub-folders to process (default: all).')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--crop', action='store_true', help='Crop a fixed window around the ring before processing.')
    parser.add_argument('--crop_x', type=int, default=1550)
    parser.add_argument('--crop_y', type=int, default=900)
    parser.add_argument('--crop_w', type=int, default=1024)
    parser.add_argument('--crop_h', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=8300.0,
                        help='Minimum sum of a [0,1]-scaled RGB patch; darker (background) patches are dropped.')
    # polar unwrapping
    parser.add_argument('--use_polar', action='store_true', help='Unwrap the ring to a (radius x angle) strip before patching.')
    parser.add_argument('--r_bins', type=int, default=512, help='Radial resolution of the polar strip.')
    parser.add_argument('--s_bins', type=int, default=1024, help='Angular resolution of the polar strip.')
    parser.add_argument('--polar_threshold', type=float, default=0.0, help='Intensity threshold used to locate the ring.')
    parser.add_argument('--signal_ratio', type=float, default=0.2,
                        help='Radii whose mean intensity exceeds signal_ratio x global mean are kept.')
    args = parser.parse_args()

    class_names = args.classes or sorted(d for d in os.listdir(args.data_dir)
                                         if os.path.isdir(os.path.join(args.data_dir, d)))
    if not class_names:
        raise SystemExit(f"[ERROR] No class sub-directories found in {args.data_dir}.")
    print(f"[INFO] classes: {class_names} | encoder: {args.encoder} | polar: {args.use_polar}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = build_model(encoder_name=args.encoder, num_classes=len(class_names), in_channels=3,
                          pretrained=True, freeze_encoder=True).encoder.to(device).eval()

    out_root = os.path.join(args.output_dir, 'polar', args.encoder) if args.use_polar \
        else os.path.join(args.output_dir, args.encoder)
    for name in class_names:
        os.makedirs(os.path.join(out_root, name), exist_ok=True)

    dataset = RingBeamDataset(root_path=args.data_dir, target_size=None, classes=class_names, crop=args.crop,
                              crop_coords=(args.crop_x, args.crop_y, args.crop_w, args.crop_h))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    with torch.no_grad():
        for img, label, fname in tqdm(loader, desc="Extracting patch features"):
            label, fname = label.item(), fname[0]
            img = img.squeeze(0).squeeze(0).numpy()
            if args.use_polar:
                img = unwrap_ring(img, args.r_bins, args.s_bins, args.polar_threshold, args.signal_ratio)

            patches = grid_patches(to_rgb(img), args.patch_size, args.stride)
            if len(patches) == 0:
                continue
            batch = torch.from_numpy(patches).float().permute(0, 3, 1, 2).div(255.0).to(device)
            batch = batch[batch.sum(dim=(1, 2, 3)) >= args.threshold]
            if batch.shape[0] == 0:
                print(f"[WARN] all patches filtered out for {fname}")
                continue

            feats = encoder(batch).flatten(1).cpu()
            save_path = os.path.join(out_root, class_names[label], f"{os.path.splitext(fname)[0]}_features.pt")
            torch.save({'features': feats, 'label': label}, save_path)

    print(f"[INFO] done -> {out_root}")


if __name__ == '__main__':
    main()
