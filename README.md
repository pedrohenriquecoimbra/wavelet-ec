# WaveletEC Module

WaveletEC is a Python module designed for processing and analyzing Eddy Covariance data using wavelet transforms. This module provides functionalities to handle EddyPro setup files, perform wavelet analysis, and integrate results.

[![DOI](https://zenodo.org/badge/DOI/10.1016/j.agrformet.2025.110684.svg)](https://doi.org/10.1016/j.agrformet.2025.110684)

[![DOI](https://zenodo.org/badge/786866970.svg)](https://zenodo.org/doi/10.5281/zenodo.11071327)

## Citation

Pedro H H Coimbra, Benjamin Loubet, Olivier Laurent, Matthias Mauder, Bernard Heinesch, Jonathan Bitton, Nicolas Delpierre, Daniel Berveiller, Jérémie Depuydt, Pauline Buysse. Evaluation of a novel approach to partitioning respiration and photosynthesis using eddy covariance, wavelets and conditional sampling. https://doi.org/10.1016/j.agrformet.2025.110684

\* corresponding author: pedro-henrique.herig-coimbra@inrae.fr

## Installation
### From pip

To install the WaveletEC module, use [pip](https://pypi.org/project/waveletec/):

Recommended step (optional):
```bash
conda create -n wavec
conda activate wavec
```
Install library:
```bash
pip install waveletec
```

### From git

To install the WaveletEC module, clone the repository and install the required dependencies:

```bash
git clone https://github.com/pedrohenriquecoimbra/wavelet-ec
cd waveletec
```
Then one of the follwoing options:
- ```pip install -r requirements.txt```
- ```conda create -n wavec --file requirements.txt```
- ```conda create -f environment.yml```


## Usage

### (Recommended) Using EddyPro

1. Make sure [EddyPro®](https://www.licor.com/support/EddyPro/software.html) is installed.
1. Run EddyPro, saving level 6 raw data.
   - go to Advanced Settings (top menu) > Output Files (left menu) > Processed raw data (bottom);
   - select Time series on "level 6 (after time lag compensation)";
   - select all variables;
   - proceed as usual running on "Advanced Mode".
1. Run _waveletec_:\
    This requires ```pip install waveletec```
    ```python
    import waveletec
    waveletec.run_from_eddypro("PATH/TO/EDDYPRO/SETUP", ...):
    ```
    Or directly in the command line:
    ```bash
    cd PATH/TO/WAVELETEC/LIBRARY
    python -m waveletec.handler --help
    python -m waveletec.handler [-args...]
    ```

### Running _waveletec_

This requires ```pip install waveletec```.
```python
import waveletec
```
Then
```python
waveletec.process(...)
```
or directly run from a DataFrame or dictionary:
```python
data = ...
waveletec.main(data, ...)
```

### Saving Results

If you're using ```waveletec.process```, just pass the ```output_folderpath``` parameter like this: ```waveletec.process(..., output_folderpath='PATH/TO/OUTPUT/FOLDER')```. This will create a new subfolder called 'wavelet_full_cospectra' for you, and you'll find the sum of all your files neatly stored in the output_folderpath you specified.

Prefer using ```waveletec.main```? No problem! Just pass the ```output_kwargs``` parameter like this: ```waveletec.main(data, ..., output_kwargs={'output_path': 'PATH/TO/OUTPUT/FOLDER'})```.


### Output Format

The output file for the wavelet-based (co)spectra analysis is structured as follows:

```cs
1   wavelet_based_(co)spectra
2   --------------------------------------------------------------
3   TIMESTAMP_START = 2022-05-13 00:00:00
4   TIMESTAMP_END = 2022-05-13 00:30:00
5   N: 133
6   TIME_BUFFER [min] = nan
7   frequency [Hz]
8   y-axis -> nan
9   mother_wavelet -> dwt
10  acquisition_frequency [Hz] = 20.0
11  averaging_interval [Min] = 30min
12  natural_frequency,variable,value
13  3.814697265625e-05,co2,417.0002460141172
.   ...,...,...
.   5.0,co2,1.708199383374261e-05
.   10.0,co2,1.7947312058017124e-07
.   ...,...,...
```

#### Explanation of Fields:

- **`wavelet_based_(co)spectra`**: Indicates that the data pertains to wavelet-based (co)spectra analysis.

- **`TIMESTAMP_START` and `TIMESTAMP_END`**: These fields specify the start and end times of the data collection period.

- **`N`**: Represents the number of data points or samples included in the analysis.

- **`TIME_BUFFER [min]`**: Indicates the time buffer in minutes. A value of 0 means no buffer was applied.

- **`frequency [Hz]`**: Specifies the frequency of the data is in Hertz.

- **`y-axis_->_wavelet_coefficient_*_`**: Indicates that the y-axis represents the wavelet coefficients.

- **`mother_wavelet -> dwt`**: Specifies the type of mother wavelet used in the analysis, in this case, `dwt` (Discrete Wavelet Transform).

- **`acquisition_frequency [Hz]`**: The frequency at which data was acquired, given in Hertz.

- **`averaging_interval [Min]`**: The interval over which the data was averaged, specified in minutes.

- **`natural_frequency,variable,value`**: This line contains the actual data points:
  - **`natural_frequency`**: The frequency at which the data point was recorded.
  - **`variable`**: The variable being measured, such as `co2`.
  - **`value`**: The value of the variable at the specified natural frequency.


## Example

For an example follow the [launcher_sample.ipynb](https://github.com/pedrohenriquecoimbra/wavelete-ec/blob/latest/sample/FR-Gri_20220514/launcher_sample.ipynb) file in [sample](https://github.com/pedrohenriquecoimbra/wavelete-ec/blob/latest/sample/).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
