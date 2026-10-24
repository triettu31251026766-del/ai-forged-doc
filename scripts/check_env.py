import platform
import sys


def main():
    print("=" * 50)
    print("Python :", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("=" * 50)

    try:
        import torch
    except Exception as e:
        print("Khong import duoc torch:", e)
        return

    print("Torch      :", torch.__version__)
    print("CUDA build :", torch.version.cuda)
    print("CUDA avail :", torch.cuda.is_available())

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        print("Device     :", torch.cuda.get_device_name(0))
        print("Capability :", cap)
        print("VRAM (GB)  :", round(props.total_memory / 1e9, 2))
        if cap >= (12, 0):
            print("Luu y: GPU Blackwell (sm_120) -> can torch build CUDA 12.8+.")
    else:
        print("Khong thay GPU CUDA. Kiem tra driver NVIDIA / ban torch cai dat.")


if __name__ == "__main__":
    main()
