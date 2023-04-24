# CS 4824: Machine Learning Final Project
## Team 1: Timothy Wu, Mathew Carter, Manasi Peta
The purpose of our project is to compare the performance of two popular supervised machine learning algorithms to find 
out which algorithm performs better. The two algorithms that we will be looking at are the Naive Bayes algorithm and the
Multilayer Perceptron (Simple Neural Network) algorithm. This markdown file will be used to explain step-by-step on how
to train and test each model.

### Data format
In order to use our models, the data must be formatted in a very specific way. The format of the data must match that of
the provided datasets for the Handwritten Digit Recognition that was supplied on Canvas. The format of the dataset is as
follows:

| label | pixel0 | pixel1 | pixel2 | ... | pixel783 |
|-------|--------|--------|--------|-----|----------|
| 8 | 0 | 0 | 0 | ... | 244      |
| 5 | 0 | 133 | 0 | ... | 54 |
| ... | ... | ... | ... | ... | ... |

The first column represents the label of the digit which is a number from 0-9. All subsequent columns titled `pixel###`
represent the grey scaled color value of the pixel which is a number between 0-255. 0 means that the pixel is completely
black and 255 means that the pixel is completely white.

### Training the Naive Bayes Model
To train our Naive Bayes model yourself, use the `naive_bayes_generate_model()` function that is located under the
`model_writing_nb.py` file. The function takes in a path to the .csv file that should be used for training the model.
The .csv file should be in the format as described above. The model will write its parameters to two .csv files called
`prior_log_prob.csv` and `conditional_log_prob.csv`. Both of these .csv files will be used in the testing function.

### Testing the Naive Bayes Model
To test our Naive Bayes model yourself, use the `naive_bayes_predict_using_model()` function that is located under the
`model_writing_nb.py` file. The function takes in a path to the .csv file that should be used for testing the model.
The .csv file should be in the format as described above. The function will return the accuracy of the model as a number
between 0 and 1.

### Training the Multilayer Perceptron Model
To train our Multilayer Perceptron model yourself, use the `multilayer_perceptron_generate_network()` function that is
located under the `model_writing_mlp.py` file. The function takes in a path to the .csv file that should be used for
training the model. The .csv file should be in the format as described above. The model will write its parameters to
four .csv files called `hidden_to_output_layer_bias_terms.csv`, `hidden_to_output_layer_weights.csv`,
`input_to_hidden_layer_bias_terms.csv`, and `input_to_hidden_layer_weights.csv`. All four of these .csv file will be 
used in the testing function.

### Testing the Multilayer Perceptron Model
To test our Multilayer Perceptron model yourself, use the `mlp_output()` function that is located under the
`model_writing_mlp.py` file. The function takes in a path to the .csv file that should be used for testing the model.
The .csv file should be in the format as described above. The function will return the accuracy of the model as a number
between 0 and 1.