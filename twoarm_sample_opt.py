import pathlib

import ipdb
import typer


def main(arm2_sol_path: pathlib.Path, arm1_sol_path: pathlib.Path):
    ...


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)()
