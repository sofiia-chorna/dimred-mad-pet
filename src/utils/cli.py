import click


def dataset(func):
    return click.option(
        "--dataset",
        default="mad",
        help="Name of the dataset: 'mad' or 'mptraj'",
        type=str,
    )(func)
