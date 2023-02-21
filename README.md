# Hierarchical-Hyperplane Kernels for Actively Learning Gaussian Process Models of Nonstationary Systems

This code is belonging to the AISTATS 2023 paper called "Hierarchical-Hyperplane Kernels for Actively Learning Gaussian Process Models of Nonstationary Systems". It can be used as a library to utilize our method/kernel and to run our algorithm with different configurations.

## Setup

After cloning the repo switch to the base folder and build the `conda` environment and the package via
```buildoutcfg
conda env create --file environment.yml
conda activate hhk
pip install -e .
```

## General Usage
Our method is build on top of gpflow. Our proposed Hierarchical-Kyperplane-Kernel class inherits from `gpflow.kernels.Kernel`. It can therefore directly be used in gpflow. If one wants to use it in this repo we offer the following setup. We use the factory patern and config classes to build all respective obejcts with the right configuration. An instance of the kernel can be easily retrieved via:
```
kernel_config = HHKFourLocalDefaultConfig(input_dimension=2)
kernel = KernelFactory.build(kernel_config)
```
For all HHK configs see file `hhk_configs.py`.
In case a GP should be configured with our kernel, we also provide a wrapper class that can be initiated via
```
model_config = GPModelWithNoisePriorConfig(kernel_config=kernel_config)
model = ModelFactory.build(model_config)
```
For the standard GP model configs see file `gp_model_config.py` and for the fully-Bayesian GP model see file `gp_model_marginalized_config.py`.
Inference for a given dataset with np.arrays `x_data` and `y_data` and test points `x_test` can than be done via
```
model.infer(x_data,y_data)
pred_mu_test,pred_sigma_test = model.predictive_dist(x_test)
```
To test our kernel in the active learning setting we provide the `main.py` script that allows the configuration of the different run settings that are used in the paper.

## License

Hierarchical-Hyperplane-Kernels is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.
