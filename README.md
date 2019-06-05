# Numpy-CNN
A numpy-only implementation of a Convolutional Neural Network, from the ground up.

## Training the network 
Train the network using the following command: 

`$ python train_cnn.py '<file_name>.pkl'`

Replace `<file_name>` with whatever file name you would like. The terminal should display the following progress bar to indicate the training progress, as well as the cost for the current training batch:

After the CNN has finished training, a .pkl file containing the network's parameters is saved to the directory where the script was run.

I included the trained params in the GitHub repo under the name `params.pkl` . To use the pretrained params when measuring the network's performance, replace `<file_name>` with params.pkl.

## Measuring Performance
To measure the network's accuracy, run the following command in the terminal:

`$ python measure_performance.py '<file_name>.pkl'`

This command will use the trained parameters to run predictions on all 10,000 digits in the test dataset. After all predictions are made, a value displaying the network's accuracy will appear in the command prompt.
