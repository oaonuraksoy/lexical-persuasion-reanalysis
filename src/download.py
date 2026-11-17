"""Download the two public datasets (Parquet exports from Hugging Face) into data/."""
from pathlib import Path
from urllib.request import urlretrieve

DATA = Path(__file__).resolve().parents[1] / "data"
FILES = {
    "persuasion.parquet":
        "https://huggingface.co/datasets/Anthropic/persuasion/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "geobench_test.parquet":
        "https://huggingface.co/datasets/GEO-Optim/geo-bench/resolve/refs%2Fconvert%2Fparquet/test/test/0000.parquet",
}

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, url in FILES.items():
        target = DATA / name
        if target.exists():
            print("exists:", target)
            continue
        print("downloading", name)
        urlretrieve(url, target)
