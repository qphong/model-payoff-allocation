# Model and Payoff Allocation

To run the allocation given the characteristic function: `run_allocation.py`.

For MNIST dataset, the characteristic function is generated from `testColabModel.py` which uses `colabModel.py`. Note that the current saving format is incorrect for `testColabModel.py`, so we correct it with `mnist_training/convert_new_format.py`.

## Requirements
* python >= 3.6
* tensorflow >= 2.4.1
* numpy >= 1.19.5
