import argparse
import glob
import os

from PIL import Image

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def first_file(dirpath):
    for e in EXTS:
        files = sorted(glob.glob(os.path.join(dirpath, "**", "*" + e), recursive=True))
        if files:
            return files[0]
    files = sorted(glob.glob(os.path.join(dirpath, "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    return files[0] if files else None


def main():
    ap = argparse.ArgumentParser(description="Kiem tra nhanh 1 cap anh + mask")
    ap.add_argument("--images", required=True, help="Thu muc chua anh")
    ap.add_argument("--masks", required=True, help="Thu muc chua mask")
    args = ap.parse_args()

    img_path = first_file(args.images)
    mask_path = first_file(args.masks)

    print("Image:", img_path)
    print("Mask :", mask_path)

    if not img_path or not mask_path:
        print("Khong tim thay file. Kiem tra lai duong dan.")
        return

    im = Image.open(img_path)
    mk = Image.open(mask_path)
    print("Image size/mode:", im.size, im.mode)
    print("Mask  size/mode:", mk.size, mk.mode)

    if im.size != mk.size:
        print("CANH BAO: kich thuoc anh va mask KHONG khop.")
    else:
        print("OK: anh va mask cung kich thuoc.")


if __name__ == "__main__":
    main()
