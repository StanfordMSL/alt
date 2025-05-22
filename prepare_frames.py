import os
import cv2
import argparse

def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"{idx:06d}.png"), frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames from {video_path} → {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="Root folder containing per‐episode subdirs with 1.mp4 & 3.mp4")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Where to write frames (mirrors episodes subdirs)")
    args = parser.parse_args()

    for ep in os.listdir(args.videos_dir):
        ep_dir = os.path.join(args.videos_dir, ep)
        if not os.path.isdir(ep_dir): continue
        hand_in  = os.path.join(ep_dir, "1.mp4")
        third_in = os.path.join(ep_dir, "3.mp4")
        extract_frames(hand_in,  os.path.join(args.output_root, ep, "hand"))
        extract_frames(third_in, os.path.join(args.output_root, ep, "third"))
