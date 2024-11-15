
> In companies, they require to write a report, so it is good if I start doing so!!!

- Improve this report

### Summary
- Aim: Classification of arrhythmias from single ECG signals
- Dataset: single ECG signals
- Data preprocessing: Normalization
- Results:


#### LR scheduler
I've decided to use the LR scheduler since I saw in the plot of the loss over the epochs that the model didn't converge completely but had an oscillatory behavior.
- I've decided to reduce the LR when there was a plateau



# Project specifications

## Data Exploratory Analysis
- Missing values
- Statistics
- Class distribution in train and test sets

## Data preprocessing
- Split in training, validation and test sets
- Normalization
- Class distribution in training, validation and test

## Models
- CNN
- RNN
- Transfer Learning
  -  I can perform **Transfer Learning**, using a **pretrained model** 
	- Since nowadays are used, I can try the **EfficientNetV2**

- weighted loss

### Layers
- Batch Normalization
- Dropout 
- L2 regularization
- Max Pooling

### Hyperparameter tuning
- Define configuration of the sweep (method, name of the sweep, metrics to minimize and hyperparameters to optimize)
  - Perform at leat 40-50 runs on sweeps

### Performance Evaluation
- Classification report
- Confusion matrix
- ROC curve
- precision-recall curve

### Error analysis

## Model comparison
- Barplot comparison
- Comparison of the average of the metrics (precision, recall and F1 score)