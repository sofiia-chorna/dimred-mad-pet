import click

from metatrain.experimental.nativepet import NativePET

from src.utils.consts import DEVICE, LFF_OUTPUT_FOLDER, DATASET_FOLDER, TYPE
from src.get_llfs import run_get_llfs


@click.group()
def main():
    pass


@main.command()
def get_llfs():
    model = NativePET.load_checkpoint("pet-mad-latest.ckpt").eval().to(DEVICE)
    
    run_get_llfs(model, DATASET_FOLDER, LFF_OUTPUT_FOLDER, TYPE)

if __name__ == "__main__":
    main()
