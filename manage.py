import os
from sys import exit, argv
from subprocess import run


def setup_venv():
    create_venv = "python3 -m venv .venv"

    try:
        if not os.path.exists(os.path.abspath(".venv")):
            run(create_venv, shell=True)
    except Exception as err:
        print(f'Check if Python is installed and on the PATH.\n\n{err}')
        exit(1)
    return


def setup_requirements():
    upgrade_deps = "pip install --upgrade pip setuptools wheel"
    install_piptools = "python3 -m pip install pip-tools pre-commit"
    install_requirements = "pip-sync requirements.txt"

    try:
        run(upgrade_deps, shell=True)
    except Exception as err:
        print(err)
        exit(1)

    try:
        run(install_piptools, shell=True)
    except Exception as err:
        print(err)
        exit(1)

    try:
        run(install_requirements, shell=True)
    except Exception as err:
        print(err)
        exit(1)

    return


def main(arg: str):
    if (arg == "setup"):
        setup_venv()
    elif (arg == "install"):
        setup_requirements()
    else:
        raise Exception("Argument is not valid!")
    return


if __name__ == "__main__":
    main(argv[1])
