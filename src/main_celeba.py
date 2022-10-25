from experiments.celeba import CelebaExperiment
from run_epochs import run_epochs


def main():
    celeb = CelebaExperiment()
    celeb.init()
    run_epochs(celeb)


if __name__ == "__main__":
    main()
