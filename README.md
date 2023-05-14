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

## To-do

- [ ] Parameterize hyperparameters for easier command-line usage

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
2. Edit hyper-parameters and settings in ```main.py```

3. navigate to any file and run it.   
 ```bash
python main.py
```


