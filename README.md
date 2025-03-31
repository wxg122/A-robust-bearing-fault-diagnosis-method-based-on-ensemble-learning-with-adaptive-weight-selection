# A-robust-bearing-fault-diagnosis-method-based-on-ensemble-learning-with-adaptive-weight-selection
The code for paper A robust bearing fault diagnosis method based on ensemble learning with adaptive weight selection
(1)You should install all the package in the requirement.txt;
(2)Data processing. The CWRU data and HIT data are all under the dataset file. You can utilize the data_process.py and cwt_process.py to obtain the CWT data. It should be noted that the original HIT data should utilize the HIT_data_extract.py to extract the different labels under .npy file.
(3) Data denoising. You should use the RCDAE.py to denoise the data.
(4)model_train.py and model_test.py are used to train and test the model.
(5)You should pay attention to the address of the file in the code.
