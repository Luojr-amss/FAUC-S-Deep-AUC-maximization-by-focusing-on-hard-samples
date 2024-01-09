FAUC-S: Deep AUC maximization by focusing on hard samples
---

[DOI: 10.1016/j.neucom.2023.127172](https://doi.org/10.1016/j.neucom.2023.127172)

Required Package Version
---------

python 3.8.0

pytorch 1.7.1

numpy 1.22.4

tensorflow 2.8.0

Datasets
---------
Four benchmark image classification datasets, namely CATvsDOG(C2), CIFAR10(C10), CIFAR100(C100), and STL10, are selected for the experiments.

Model and Environment
---------
In the experiment, ResNet20 is chosen as the network structure. We use a batch size = 128 and train for a total of 200 epochs, and a 9:1 train/validation split to conduct cross-valuation for tuning parameters.
The weight decay is set to 10−4 for all experiments. 

The imbalance ratio is manually set to 0.1 and 0.01. The experiment is run five times with different random seeds for all benchmark datasets. The mean and standard deviations(std) of test AUC are calculated. 


Click on the FAUC-S main program to run.
---------

Citation
---------
If you find this work useful in your work, please cite the following papers

(https://github.com/Luojr-amss/FAUC-S/files/13871244/1-s2.0-S092523122301295X-main.pdf):
```
@article{
    Xu2024FAUC-S,
    title={FAUC-S:Deep AUC maximization by focusing on hard samples},
    author={Shoukun Xu, Yanrui Ding, Yanhao Wang, Junru Luo},
    journal={Neurocomputing},
    volume = {571},
    pages = {127172},
    year={2024},
}
 ```

Contact
----------
For any technical questions, please open a new issue in the Github. If you have any other questions, please contact us @ [Junru Luo][luojunru@cczu.edu.cn] and [Yanrui Ding][S21150812050@smail.cczu.edu.cn].


