# Citation

Pedro H H Coimbra, Benjamin Loubet, Olivier Laurent, Matthias Mauder, Bernard Heinesch, Jonathan Bitton, Jeremie Depuydt, Pauline Buysse. Improvement of CO2 Flux Quality Through Wavelet-Based Eddy Covariance: A New Method for Partitioning Respiration and Photosynthesis. http://dx.doi.org/10.2139/ssrn.4642939

\* corresponding author: pedro-henrique.herig-coimbra@inrae.fr


# Getting started

1. Setup python.\
(optional) Create python environment, with anaconda prompt run `conda create -n wavec`\
(optional) Activate new environement, `activate wavec`\
Install python library, `pip install waveletec`

2. Run EddyPro, saving level 6 raw data. \
To do this go in Advanced Settings (top menu) > Output Files (left menu) > Processed raw data (bottom);\
Then select Time series on "level 6 (after time lag compensation)";\
Select all variables;\
Proceed as usual running on "Advanced Mode".

3. Follow launcher.ipynb

#### If directly cloning github

1. Setup python.\
(option 1) install anaconda, and run `conda create -n wavec --file requirements.txt`\
(option 2) install anaconda, and run `conda create -f environment.yml`

# Example

For an example follow the [launcher_sample.ipynb](https://github.com/pedrohenriquecoimbra/wavelete-ec/blob/latest/sample/FR-Gri_20220514/launcher_sample.ipynb) file in folder sample\FR-Gri_20220514.
