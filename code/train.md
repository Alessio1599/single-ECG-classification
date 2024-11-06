## Here I will explain better the training process
In order to train the model we have to deine the architecture (number of layers, etc.).
- particular layers to avoid overfitting
- activation functions

In addition to that we have to define
- optimizer
- Learning rate
  - warmup of the LR
- loss function
- number of epochs

## I can check on Stack Overflow, to see how they usually implement the learning rate schedule !!!

## Problem that I had with callbacks
### I WAS HAVING DIFFICULTIES IN DEFINING CALLBACKS..

The issue is that the LearningRateScheduler callback resets the learning rate every epoch, which overrides the adjustments made by the ReduceLROnPlateau callback. This conflict can be resolved by carefully managing the learning rate such that the LearningRateScheduler only modifies the learning rate during the warm-up period and leaves it unchanged afterward.
