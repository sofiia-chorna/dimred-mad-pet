import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = ["train", "val", "test"]
BATCH_SIZE = 64

DATASET_FOLDER = "data/v1.5"
TYPE = "cleaned-pbc"
OUTPUT_FOLDER = f"{DATASET_FOLDER}_{TYPE}_merged"

LFF_OUTPUT_FOLDER = f"{DATASET_FOLDER}_{TYPE}_llfs"
