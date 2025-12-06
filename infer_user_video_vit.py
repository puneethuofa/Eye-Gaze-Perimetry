#!/usr/bin/env python3
import argparse, csv
import cv2, torch, timm
import torchvision.transforms as T
from PIL import Image
import math

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

def load_model(ckpt_path, model_name="vit_tiny_patch16_224", img_size=96):
    extra_kwargs = {}
    if "vit" in model_name.lower():
        extra_kwargs["img_size"] = img_size

    model = timm.create_model(model_name, pretrained=False, num_classes=2, **extra_kwargs)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # From train_vit_mpiigaze.py we saved "model_state"
    state = ckpt.get("model_state", ckpt.get("model"))
    if state is None:
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain model_state or model keys.")
    model.load_state_dict(state, strict=False)

    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--target-size", type=int, default=96)
    ap.add_argument("--model", default="vit_tiny_patch16_224")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, args.model, img_size=args.target_size).to(device)

    # Same normalization as training
    tfm = T.Compose([
        T.Resize((args.target_size, args.target_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_sec","yaw_deg","pitch_deg"])
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # BGR -> RGB PIL
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                y_rad = model(x)[0]   # [yaw_rad, pitch_rad]
            y_deg = y_rad * RAD2DEG
            t = i * (1.0 / float(args.fps))
            wr.writerow([f"{t:.6f}", f"{y_deg[0].item():.6f}", f"{y_deg[1].item():.6f}"])
            i += 1

    cap.release()
    print(f"[infer] wrote {args.out}")

if __name__ == "__main__":
    main()

