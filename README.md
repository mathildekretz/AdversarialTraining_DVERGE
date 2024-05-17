# Adversarial Robustness through Training using the DVERGE Method - DataLabAssignement3

## IASD Master Program 2023/2024 - PSL Research University

### About this project

This project is the third homework assignment for the Data Science Lab class of the IASD (Artificial Intelligence, Systems and Data) Master Program 2023/2024, at PSL Research University (Université PSL).

*The project achieved the following objectives:*
- Improved the adversarial robustness of a CIFAR10 classification model using the DVERGE method, derived from vulnerability diversification. This non-adversarial training effectively isolated adversarial vulnerabilities, resulting in heightened ensemble robustness while preserving accuracy with clean data.
- Explored the DVERGE training procedure, showcasing its efficacy in fortifying the model against adversarial attacks without compromising accuracy on clean data.
- Investigated the integration of adversarial training into our methodology, uncovering additional enhancements in robustness.

## General Information

The report can be viewed in the [report.pdf](report.pdf) file. It answers to the instructions given in the [assignment_3_slides_instructions.pdf](assignment_3_slides_instructions.pdf) file provided by the professors.

The rest of the instructions can be found below. If you want to copy and recreate this project, or test it for yourself, some important information to know.

**requirements.txt**
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt

**train and validate vanilla models on clean dataset**
  > ./model.py
    
**train and validate vanilla models on adversarial data and attacks**
  > ./adv_model.py
    
**train and validate DVERGE models**
  > ./dverge.py
    
**test the vanilla or DVERGE pretrained models**
  > ./test_project.py

**test the ensemble model**
  > ./model_ensemble_test.py

**modifying the parameters**
  
    All paramaters are set in the config.json file, loaded within the functions each time it is necessary.
    This includes: epochs, batch_size, lr, different epsilons, different alphas ...
    Except: 
      - testing arguments, in the parser of the main function of the test_project.py file
      - the adversarial traing mode, in the main function of the dverge.py file (see below)

**set the DVERGE model into adversarial training (DVERGE+AdvT method)**

    This parameter is defined in the main function of the dverge.py file (line 169) and is set to True if the training is DVERGE+AdvT, and False if the training is DVERGE.

**load the module project and test it as close as it will be tested on the testing plateform**
  > ./test_project.py

**models**
Push the minimal amount of models in the folder *models*.

---

### Acknowledgments 

This project was made possible with the guidance and support of the following :

- **Prof. Benjamin Negrevergne**
  - Professor at *Université Paris-Dauphine, PSL*
  - Researcher in the *MILES Team* at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
  - Co-director of the IASD Master Program with Olivier Cappé

- **Alexandre Vérine**
  - PhD candidate at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
 
This project was a group project, and was made possible thanks to the collaboration of :

- **Mathilde Kretz**, *IASD Master Program 2023/2024 student, at PSL Research University*
- **Alexandre Ngau**, *IASD Master Program 2023/2024 student, at PSL Research University*

We would like to acknowledge the contribution of the codebase used in this project, which was inspired by the implementation available at https://github.com/zjysteven/DVERGE. The original code is associated with the work presented in the paper titled :
H. Yang, J. Zhang, H. Dong, N. Inkawhich, A. Gardner, A. Touchet, W. Wilkes, H. Berry, and H. Li, “Dverge:
Diversifying vulnerabilities for enhanced robust generation of ensembles,” 2020.
https://arxiv.org/pdf/2009.14720.pdf

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

**Note:**

This project is part of ongoing research and is subject to certain restrictions. Please refer to the license section and the [LICENSE.md](LICENSE.md) file for details on how to use this code for research purposes.
