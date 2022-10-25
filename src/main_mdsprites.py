from experiments.mdsprites.experiment import MdSpritesExperiment
from run_epochs import run_epochs


def main():
    mdsprites = MdSpritesExperiment()
    mdsprites.init()
    run_epochs(mdsprites)


if __name__ == "__main__":
    main()
