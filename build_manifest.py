#!/usr/bin/env python3
import argparse, os, sys, re, csv
from pathlib import Path

def find_ann_dir(root: Path) -> Path:
    cands = [
        root / "Annotation Subset",
        root / "annotation subset",
        root,
    ]
    for p in cands:
        if any((p / f"p{n:02d}.txt").exists() for n in range(15)):
            return p
    return root

def iter_day_dirs(root: Path):
    """Stream a map dayXX -> list of directories that are named dayXX (low memory)."""
    day_map = {}
    for dirpath, dirnames, _filenames in os.walk(root):
        for d in dirnames[:]:
            if d.startswith("day") and d[3:].isdigit():
                day_map.setdefault(d, []).append(Path(dirpath) / d)
    return day_map

def clean_img_token(tok: str) -> str:
    tok = tok.strip().strip(",").replace("\\", "/")
    if tok.endswith(".jpg.jpg"):
        tok = tok[:-4]
    # keep from 'day' forward if present
    i = tok.find("day")
    if i >= 0:
        tok = tok[i:]
    return tok

_num = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

def parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    toks = re.split(r"[,\s]+", line)
    # locate image-ish token
    img_tok = None
    for t in toks:
        tl = t.lower()
        if "day" in tl and (tl.endswith(".jpg") or tl.endswith(".jpeg") or tl.endswith(".png")):
            img_tok = t
            break
    if img_tok is None:
        for t in toks:
            if ".jpg" in t.lower():
                img_tok = t
                break
    if img_tok is None:
        return None

    nums = [t for t in toks if _num.match(t)]
    if len(nums) < 2:
        return None
    yaw_deg = float(nums[-2]); pitch_deg = float(nums[-1])

    img_suffix = clean_img_token(img_tok)  # e.g., day13/0203.jpg
    day_m = re.search(r"(day\d{1,2})", img_suffix)
    day = day_m.group(1) if day_m else ""
    subj_m = re.search(r"\bp(\d{2})\b", line)
    subj = f"p{subj_m.group(1)}" if subj_m else ""
    name = img_suffix
    return img_suffix, yaw_deg, pitch_deg, subj, day, name

def resolve_path(img_suffix: str, day_dirs: dict[str, list[Path]]) -> str | None:
    """Try joining the file onto every directory named that day; no global image index."""
    img_suffix = img_suffix.lstrip("./")
    day_m = re.match(r"^(day\d{1,2})/(.+)$", img_suffix)
    if not day_m:
        return None
    day = day_m.group(1); fname = day_m.group(2)
    dirs = day_dirs.get(day, [])
    # try exact, and a few small variants
    candidates = []
    for d in dirs:
        candidates.append(d / fname)
        # sometimes files live one level deeper
        # e.g., p00/day13/0203.jpg vs <something>/day13/0203.jpg — handled by day_dirs already
    # try de-duped extension fix
    if fname.endswith(".jpg.jpg"):
        f2 = fname[:-4]
        for d in dirs:
            candidates.append(d / f2)
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", default="img,yaw_deg,pitch_deg,subj,day,name")
    args = ap.parse_args()

    root = Path(args.dataset_root).resolve()
    ann_dir = find_ann_dir(root)
    if not ann_dir.exists():
        print(f"[error] annotation dir not found under {root}", file=sys.stderr); sys.exit(2)

    # Build cheap map of dayXX directories only (much smaller than indexing all images)
    print(f"[index] walking for day folders under: {root}")
    day_dirs = iter_day_dirs(root)
    total_day_dirs = sum(len(v) for v in day_dirs.values())
    print(f"[index] discovered {len(day_dirs)} unique days spanning {total_day_dirs} directories.")

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)

    kept = 0; total_lines = 0
    with out_p.open("w", newline="") as f:
        ww = csv.writer(f); ww.writerow(args.cols.split(","))
        for i in range(15):
            txt = ann_dir / f"p{i:02d}.txt"
            if not txt.exists(): continue
            lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            kept_this = 0
            for ln in lines:
                total_lines += 1
                rec = parse_line(ln)
                if not rec: continue
                img_suffix, yaw_deg, pitch_deg, subj, day, name = rec
                full = resolve_path(img_suffix, day_dirs)
                if full:
                    ww.writerow([full, yaw_deg, pitch_deg, subj, day, name])
                    kept += 1; kept_this += 1
            print(f"[manifest] p{i:02d}.txt: parsed {len(lines)} lines, kept {kept_this} rows")
    print(f"[manifest] wrote {out_p} with {kept} rows (from {total_lines} lines)")

if __name__ == "__main__":
    main()
