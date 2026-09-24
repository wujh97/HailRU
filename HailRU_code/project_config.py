from pathlib import Path
from collections.abc import Mapping
import os
import pickle
import re

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CATALOG_ROOT = DATA_ROOT / "catalogs"
SAMPLE_ROOT = DATA_ROOT / "samples"
RADAR_DOC_ROOT = DATA_ROOT / "docs" / "RADAR_Format_DOC"
NWP_DOC_ROOT = DATA_ROOT / "docs" / "NWP_Format_DOC"
WEIGHTS_ROOT = ROOT / "weights"
OUTPUT_ROOT = ROOT / "outputs"
STATISTICS_ROOT = OUTPUT_ROOT / "statistics"
CITY_BOUNDARY_PATH = DATA_ROOT / "boundaries" / "china_prefecture_WGS84.geojson"
PATH_REMAP = {}

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
DATASET_SPLIT = "test"
EVALUATION_SPLIT = "test"
CASE_SPLIT = "validation"
CASE_ID = "HA_08_06011755_00028"
CASE_INDEX = 8
CASE_TITLE = CASE_ID
RANDOM_SEED = 20260724
NUM_WORKERS = 0
DEVICES = 1
STRATEGY = "auto"
HAILRU_CHECKPOINT = WEIGHTS_ROOT / "HailRU" / "best_model.ckpt"
WEIGHTED_HAILRU_CHECKPOINT = WEIGHTS_ROOT / "HailRU_weighted" / "best_model.ckpt"
DIFF_T_CHECKPOINT = WEIGHTS_ROOT / "DIFF_T" / "best_model.ckpt"
UNET_CHECKPOINT = WEIGHTS_ROOT / "UNet" / "best_model.ckpt"


def resolve_path(value):
    text = os.fspath(value).replace("\\", "/")
    for old, new in sorted(PATH_REMAP.items(), key=lambda pair: -len(pair[0])):
        old = str(old).replace("\\", "/").rstrip("/")
        if text == old or text.startswith(old + "/"):
            text = str(new).replace("\\", "/").rstrip("/") + text[len(old):]
            break
    if re.match(r"^[A-Za-z]:/", text) and os.name != "nt":
        raise FileNotFoundError("Set PATH_REMAP in project_config.py for: " + text)
    path = Path(text).expanduser()
    return path if path.is_absolute() else ROOT / path


def remap_paths(value):
    if isinstance(value, dict):
        return {k: remap_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [remap_paths(v) for v in value]
    if isinstance(value, tuple):
        return tuple(remap_paths(v) for v in value)
    if isinstance(value, Path):
        return str(resolve_path(value))
    if isinstance(value, str) and re.search(r"\.(npy|npz|pkl|ckpt|pt)$", value, re.I):
        return str(resolve_path(value))
    return value


def load_pickle(path):
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}. See README.md.")
    with path.open("rb") as stream:
        return remap_paths(pickle.load(stream))


def load_split_samples(split):
    directory = CATALOG_ROOT / split
    matrix_path = directory / "train_dic_matrix.pkl"
    if matrix_path.is_file():
        matrix = load_pickle(matrix_path)
        records = []
        for samples in matrix.values():
            records.extend(samples.values() if isinstance(samples, Mapping) else samples)
        x = [r["input"] for r in records]
        masks = [r["mask"] for r in records]
        y = [r["output"] for r in records]
    else:
        x = load_pickle(directory / "train_dic_x_nwp.pkl")
        masks = load_pickle(directory / "train_dic_mask_nwp.pkl")
        y = load_pickle(directory / "train_dic_y_nwp.pkl")
    if not x or not (len(x) == len(masks) == len(y)):
        raise ValueError(f"Empty or inconsistent sample lists for split {split!r}.")
    return x, masks, y


def write_sample_index(split):
    directory = SAMPLE_ROOT / split
    matrix = {}
    for file in sorted(directory.glob("*/*/input/*.npz")):
        case_id = file.parent.parent.name
        mask = file.parent.parent / "mask" / file.name
        target = file.parent.parent / "output" / file.name
        if not mask.is_file() or not target.is_file():
            raise FileNotFoundError(f"Incomplete sample: {file}")
        record = {}
        for key, path in (("input", file), ("mask", mask), ("output", target)):
            record[key] = os.path.relpath(path, ROOT).replace(os.sep, "/")
        matrix.setdefault(case_id, {})[int(file.stem)] = record
    if not matrix:
        raise ValueError(f"No complete samples were found in {directory}.")
    records = [r for samples in matrix.values() for r in samples.values()]
    destination = CATALOG_ROOT / split
    destination.mkdir(parents=True, exist_ok=True)
    items = {
        "train_dic_matrix.pkl": matrix,
        "train_dic_x_nwp.pkl": [r["input"] for r in records],
        "train_dic_mask_nwp.pkl": [r["mask"] for r in records],
        "train_dic_y_nwp.pkl": [r["output"] for r in records],
    }
    for name, value in items.items():
        with (destination / name).open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return matrix


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0
