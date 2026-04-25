import io
from typing import Iterable

from datasets import load_dataset
from PIL import Image as PILImage


def _decode_bytes(b: bytes) -> PILImage.Image:
    img = PILImage.open(io.BytesIO(b))
    img.load()
    return img


def _iter_n(it: Iterable, n: int):
    for _ in range(n):
        yield next(it)


def main():
    name = "lyl472324464/twist_subset_balanced_100k_448_multi_repo_viewerfix_rg50"
    split = "train"

    print(f"loading(streaming) {name} [{split}]")
    ds = load_dataset(name, split=split, streaming=True)

    print("\\nfeatures:")
    for k, v in ds.features.items():
        print(f"- {k}: {v}")

    img_cols = [
        k
        for k, v in ds.features.items()
        if k.startswith("observation.images.") and isinstance(v, dict) and "bytes" in v and "path" in v
    ]
    print("\\nimage columns:", img_cols)

    it = iter(ds)
    for i, ex in enumerate(_iter_n(it, 3)):
        print(f"\\nexample[{i}]")
        for k in img_cols:
            v = ex[k]
            b = v.get("bytes", None) if isinstance(v, dict) else None
            img = _decode_bytes(b)
            print(f"- {k}: decoded -> mode={img.mode} size={img.size}")

    print("\\n结论：")
    print(
        "- 这个数据集里图片列在 schema 上是 {bytes: binary, path: string}，不是 datasets.Image()。\\n"
        "- 但 bytes 是可直接解码的标准图片（上面已解码到 RGB 448x448）。\\n"
        "- 所以‘图片变成 bytes’本身不影响读图；关键在于上层库（如 LeRobot）是否会把该列 cast/解码成 PIL。\\n"
        "- 如果 LeRobot 直接按 Image feature 走 transform，那你需要先把这些列 cast 回 datasets.Image() 再喂给 LeRobot。"
    )


if __name__ == "__main__":
    main()
