# Software environment
python3.0 +,
pytorch above 1.3.1,
torchvision 0.4.1 +,
Pillow 7.1.2 and above,
pandas 1.0.3 +

# Project structure
! [Project Structure](img/18.png)

data directory: csv file of the SSE index
model directory: Model save files
dataset.py: Data loading and preprocessing classes, data normalization, splitting training and test sets, etc
Plot.py: Preanalyze the data and plot the stock movement and volume trends
evaluate.py: prediction
LSTMModel.py: Defines the LSTM model
parsermy.py: Common parameters
train.py: Model training

# How to run:

To start training the model, simply run train.py

Just run evaluate.py to start making model predictions
