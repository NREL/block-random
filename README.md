# A block-random algorithm for learning on distributed, heterogeneous data

## Python environments

The quickest way to setup the Python environment is to use `pipenv install` and then `pipenv shell`.

## Benchmarks on classification EMNIST data sets

Reproducing the benchmark results from the paper entails the following steps:

1. Download and unzip the [EMNIST data sets](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip) into `./data/emnist-dataset` folder

1. Train CNN on different data sets with `shuffled`, `sorted` and `block-random` orderings, at different batch sizes:

    ``` shell
    $ python classification-tests.py -d fashion
    $ python classification-tests.py -d digits
    $ python classification-tests.py -d letters
    $ python classification-tests.py -d byclass
    $ python classification-tests.py -d bymerge
    $ python classification-tests.py -d balanced
    $ python classification-tests.py -d mnist"
    ```
    The outputs are stored in the `batch_size_study` directory.

1. Plot the results: `$ python plot.py`


## Predicting `$\tau_{ij}$` for LES of channel flow

Reproducing the channel flow results from the paper entails the following steps:

1. Generate the filtered data from the DNS: `$ python scaling.py`. This creates `scaled.npy` in the data directory which has the filtered velocities, gradients and $\tau_ij$ terms.

1. Generate the training and test data for the various runs:
   ```
   $ python gen_data.py -o shuffled-1m -p shuffled -b 16 -n 1000000
   $ python gen_data.py -o shuffled-16 -p shuffled -b 16
   $ python gen_data.py -o block-16 -p block -b 16
   $ python gen_data.py -o sorted-16 -p sorted -b 16
   ```

1. Perform hyperparameter sweeps using the shell script:
    ``` shell
    $ sh parameter_sweeps.sh 1
    $ sh parameter_sweeps.sh 2
    $ sh parameter_sweeps.sh 3
    $ sh parameter_sweeps.sh 4
    ```

1. Plot comparisons of results: `$ python compare_runs.py -r runs`

1. Train the models using different types of algorithms: `$ sh model_runs.sh`

1. Plot a given model result: `$ python plot_run.py -r runs/${directory_name}`

## Citation for this work
```
@article{Mohan19,
    author    = {P. Mohan, M. T. Henry de Frahan, R. King, and R. W. Grout},
    title     = {A block-random algorithm for learning on distributed, heterogeneous data},
    journal   = {arXiv:1903.00091},
    year      = {2019}
}
```
