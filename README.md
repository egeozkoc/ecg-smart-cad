# ecg-smart
Repository for all ECG-SMART related code

1) Environment requirements
- requirements_linux_py312.txt: python library requirements for linux
- requirements_windows_py312.txt: python library requirements for windows

2) Preprocessing
- ecg.py: main code to run preprocessing of xml files
- get_10sec.py: helper file to open different ecg file formats, get 12-lead 10sec ECG data, filter, and remove artifacts
- get_median.py: helper file to get the average 12-lead heartbeat from the clean 10sec ECG data
- get_fiducials.py: helper file to get onset and offset of P, QRS, and T
- get_features.py: helper file to extract features from the ECG data

3) ML Models
- rf.py: code to run random forest models
- train_models.py: code to train CNN models
- ecg_models.py: CNN model architectures

4) Explainability
- TODO: update these files to newest versions