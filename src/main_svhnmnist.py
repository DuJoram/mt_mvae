from experiments.mnistsvhntext.experiment import MNISTSVHNText
from run_epochs import run_epochs

if __name__ == "__main__":
    mst = MNISTSVHNText()
    mst.init()
    run_epochs(mst)
