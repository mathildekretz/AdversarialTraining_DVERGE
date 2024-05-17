[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/vpaycMdZ)
This is a basic code repository for Assignment 3.

The repository contains a basic model and a basic training and testing
procedure. It will work on the testing-platform (but it will not
perform well against adversarial examples). The goal of the project is
to train a new model that is as robust as possible.

# Acknowledgments 

We would like to acknowledge the contribution of the codebase used in this project, which was inspired by the implementation available at https://github.com/zjysteven/DVERGE. The original code is associated with the work presented in the paper titled :
H. Yang, J. Zhang, H. Dong, N. Inkawhich, A. Gardner, A. Touchet, W. Wilkes, H. Berry, and H. Li, “Dverge:
Diversifying vulnerabilities for enhanced robust generation of ensembles,” 2020.
https://arxiv.org/pdf/2009.14720.pdf


# Basic usage

Install python dependencies with pip: 

    $ pip install -r requirements.txt

Overview:
    To train and validate vanilla models on clean dataset: 
        $ ./model.py
    
    To train and validate vanilla models on clean data and attacks:
        $ ./adv_model.py
    
    To train and validate DVERGE models:
        $ ./dverge.py
    
    To test the pre trained models:
        $ ./test_project.py

    To test the ensemble model:
        $ ./ model_ensemble_test.py

Modify the parameters:

    All paramaters are set on config.json, loaded within the functions each time it is necessary
    This includes: epochs, batch_size, lr, different epsilon, different alpha ...
    Except: testing arguments in the parser of main of test_project.py and the adversarial traing inside the main of dverge function (see below)

Set the DVERGE model into adversarial training:

    This parameter is defined in function main of dverge.py (line 169) and is set to True is the training is wanted to be adversarial or False if not.


Load the module project and test it as close as it will be tested on the testing plateform:

    $ ./test_project.py

Even safer: do it from a different directory:

    $ mkdir tmp
    $ cd /tmp
    $ ../test_project.py ../
 
# Modifying the project

You can modify anything inside this git repository, it will work as long as:

- it contains a `model.py` file in the root directory
- the `model.py` file contains a class called `Net` derived from `torch.nn.Module`
- the `Net` class has a function call `load_for_testing()` that initializes the model for testing (typically by setting the weights properly).  The default load_for_testing() loads and store weights from a model file, you will also need to make sure the repos contains a model file that can be loaded into the `Net` architecture using Net.load(model_file).
- You may modify this `README.md` file. 

# Before pushing

When you have made improvements your version of the git repository:

1. Add and commit every new/modified file to the git repository, including your model files in models/.(Check with `git status`) *DO NOT CHECK THE DATA IN PLEASE!!!!*
2. Run `test_project.py` and verify the default model file used by load_for_testing() is the model file that you actually want to use for testing on the platform. 
3. Push your last change

Note: If you want to avoid any problems, it is a good idea to make a local copy of your repos (with `git clone <repos> <repos-copy>`) and to test the project inside this local copy.

Good luck!
