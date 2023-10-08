<div align="center">    
 
# Dual-Model Active Learning Framework



<!--  
Conference   
-->   
</div>
 
## Overview   
We propose a novel Active Learning (AL) architecture to support and reduce human annotations of both labels and explanations in low-resource scenarios. Our AL architecture incorporates an explanation-generation model that can explicitly generate natural language explanations for the prediction model and for assisting humans' decision-making in real-world. For our AL framework, we design a data diversity-based AL data selection strategy that leverages the explanation annotations.

We conduct AL Simulation experiment on the e-SNLI dataset and provide code to reproduce the results. Currently, all the experiment hyper-parameters can be edited from ```main.py```, including the selection of AL data selector.

We highly suggest running the code with GPU to reduce the experiment time. 

## How to run   

0. (optional) Create a conda env for this project

1. Install dependencies   
```bash
# clone project   
git clone https://github.com/leoleoasd/dual_model_active_learning

# install project   
cd dual_model_active_learning
pip install -r requirements.txt
 ```

2. run experiment
 ```bash
python main.py --criteria uncertainty_rationale
```

```
usage: main.py [-h] [--num_iter NUM_ITER] [--num_data_per_batch NUM_DATA_PER_BATCH] [--num_epochs_rg NUM_EPOCHS_RG] [--num_epochs_p NUM_EPOCHS_P] [--learning_rate LEARNING_RATE]
               [--per_device_batch_size PER_DEVICE_BATCH_SIZE] [--criteria {random,even,even_rationale,uncertainty,uncertainty_rationale}]

optional arguments:
  -h, --help            show this help message and exit
  --num_iter NUM_ITER
  --num_data_per_batch NUM_DATA_PER_BATCH
  --num_epochs_rg NUM_EPOCHS_RG
  --num_epochs_p NUM_EPOCHS_P
  --learning_rate LEARNING_RATE
  --per_device_batch_size PER_DEVICE_BATCH_SIZE
  --criteria {random,even,even_rationale,uncertainty,uncertainty_rationale}
```

