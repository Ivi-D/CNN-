# Deep learning for Quantitative MRI using Convolutional Neural Network

Purpose: Demonstrate a method for reconstruction of multi‐dimensional MR fingerprinting (MRF) data using deep learning methods.


1) The file "CNN_model" contain the main code for training the CNN network

2) The "STFT_images" file generates the STFT time-frequency map of the raw signal.

3) The "train-test_dataframes" file creates the train and test dataframes listing the STFT map images, in order to train the CNN model using "flow_from_dataframe" function.