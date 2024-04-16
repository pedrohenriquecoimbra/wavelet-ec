# Citation

Pedro H H Coimbra, Benjamin Loubet, Olivier Laurent, Matthias Mauder, Bernard Heinesch, Jonathan Bitton, Jérémie Depuydt, Pauline Buysse. Improvement of CO2 Flux Quality Through Wavelet-Based Eddy Covariance: A New Method for Partitioning Respiration and Photosynthesis. http://dx.doi.org/10.2139/ssrn.4642939

\* corresponding author: pedro-henrique.herig-coimbra@inrae.fr


# (Recommended) step-by-step

1. Setup python.\
(suggestion) install anaconda, and run `conda create -n wavec --file requirements.txt`

2. Run EddyPro, saving level 6 raw data. \
To do this go in Advanced Settings (top menu) > Output Files (left menu) > Processed raw data (bottom);\
Then select Time series on "level 6 (after time lag compensation)";\
Select all variables;
Proceed as usual running on "Advanced Mode".

3. Follow launcher.ipynb
