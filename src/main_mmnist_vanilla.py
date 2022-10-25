from experiments.mmnist.experiment import MMNISTExperiment
from run_epochs import run_epochs

if __name__ == "__main__":
    mst = MMNISTExperiment()
    mst.init()
    run_epochs(mst)
