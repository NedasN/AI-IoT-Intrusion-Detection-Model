# Year-4-Dissertation-Project
The technical part of the Dissertation project aiming to use deep learning and the new ToN_IoT dataset to classify attacks on IoT networks.

## Model Testing
Gradient Descent Models can be testing using the [ModelTest.py](https://github.com/NedasN/AI-IoT-Intrusion-Detection-Model/blob/main/Model%20Testing/ModelTest.py) The code can be adjusted to run any of the models and can run on the whole dataset or a small 1% random subset of the data. Will print performance metrics and confusion matrix for the samples selected.

## Gradient Descent Models
There are 3 pre-trained models, there is the [multiclass model](Gradient Descent Model/MulticlassModel.pth) and 2 Binary classifyers, [One that was trained for 900 epochs](Gradient Descent Model/GradTrainedModel.pth), and [A model trained for 700 epochs](Gradient Descent Model/700EpochGrad.pth).
The training for code for these models has been included in the respective python files; [GradModel.py](Gradient Descent Model/GradModel.py) and [MulticlassModel.py](Gradient Descent Model/MulticlassModel.py).

## PSO Models
The are 2 attempted particle swarm optimised models, using 2 different libraries, both of which got stuck in a local optima and got outclassed by the gradient descent models, files for the optimisation code can be found [here](PSO Model)

## Dataset
This project used the ToN_IoT dataset available from the [UNSW Research website](https://research.unsw.edu.au/projects/toniot-datasets)
