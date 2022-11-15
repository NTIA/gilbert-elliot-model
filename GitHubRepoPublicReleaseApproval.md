# GitHub Repository Public Release Approval

### Project Name: Audio Quality Research Project
### Software Name: Gilbert-Elliot-model

The project identified above, which is contained within the repository this document is stored in, has met the following criteria for public release:

1. [X] The project, including the test criteria, meets the requirements defined in the ITS Software Development Publication Policy for making a repository public. The major pre-established criteria for publication are listed below, and the check mark next to each attests that the criterion has been met.
    * [X] Simulate errors using the Gilbert-Elliot burst model and a variety of controls as parameters
    * [X] Deteremine the most likely version of the Gilbert-Elliot model given observation.
    * [X] Determine error statistics given model parameters.
    * [X] Measure error statistics from an error pattern.
    * [X] Fit a two-state hidden Markov model for the Gilbert-Elliot model to an observed error pattern.
    * [X] Determine valid ranges for Gilbert-Elliot model controls and error statistics
2. [X] Any test data necessary for the code to function is included in this GitHub repository.
3. [X] The README.md file is complete.
4. [X] The project complies with the ITS Code Style Guide or an appropriate style guide as agreed to by the sponsor, project lead, or Supervising Division Chief.
    * This project deviates from ITS Code Style Guide primarily in `gilbert_elliot_model/symbolic_solutions.py`. This code relies on a symbolic solver and having agreement between code variable names and the variable names used by the solver is advantageous. Further these variable names are well defined throughout the documentation, are used in the accompanying NTIA technical memorandum, and are consistent with existing nomenclature in this field. 
5. [X] Approved disclaimer and licensing language has been included.

In order to complete this approval, please create a new branch, upload and commit your version of this Markdown document to that branch, then create a pull request for that branch. The following must login to GitHub and approve that pull request before the pull request can be merged and this repo made public:
* Project Lead - Steve Voran
* Supervising Division Chief or Release Authority - Shariq Ashfaq