# tensorflow

## Course Material

<details>
  <summary>Details</summary>
  
  Feel free to edit and update content

[Neural Networks Course - KPLR](https://sylacefr.sharepoint.com/:b:/s/KPLR/EbMIbw-zj7xCtf_8uuInlMABvH9HahGd-YzZgF65Rwj31g?e=UZpdCQ)

[Convolutional Neural Network](https://docs.google.com/presentation/d/1R0aAw0dkHO0FMjWc7_dNb0vYR6lXM65OJxluRZcQ9X8/edit#slide=id.p)

[Neural Networks Course (2) - KPLR](https://sylacefr.sharepoint.com/:p:/s/KPLR/EWgGzXjvPoRMlTNeiU-M8m8BTRh2AvfOpIoWV3TVeyXH6w?e=VrFcr3)  
(maybe this course is just a duplicate)

[TensorFlow Course - KPLR](https://sylacefr.sharepoint.com/:p:/s/KPLR/EeVyRhQ9BhJOmNTt2oX7KXMBluTwwm8nLCabD4bxrJKTmQ?e=iV6exk)

[TensorFlow Book](https://sylacefr.sharepoint.com/:b:/s/KPLR/EVE-w4xgJ0ZCmbTC5theHVYBVcG1SbnsTrmZ9ZVJXNwNUw?e=EmjQDJ)  
Warning : book is outdated (2017) and deals with TF V1 - Some general concepts apply but be aware that V2 had major shifts compared to V1. 
Caution is to be applied - Critical cross-reading need to be applied

[LSTM Course](https://sylacefr.sharepoint.com/:b:/s/KPLR/EV7ec_n0-NpJmelVQL1kAysBJoUSV1PejzFZxaImCc_3Xw?e=aVe4tx)

</details>

## Planning : 

<details>
  <summary>Details</summary>
  
### DAY ONE 
| Trainer Name | Topic |  Presentation | Exercice Notebook | Solution Notebook | External Ressource/example |
| ------ | ------ | ------ |  ------ | ------ | ------ |
| Mehdi | TF Basics | [TF presentation](https://sylacefr.sharepoint.com/:p:/s/KPLR/EeVyRhQ9BhJOmNTt2oX7KXMBluTwwm8nLCabD4bxrJKTmQ?e=iV6exk) | | |
| Mehdi | Tensors | | | |
| Mehdi | Variables |  |
| Mehdi | Automatic Differenciation |  |
| Mehdi | Graphs |  |
| Mehdi | Modules | | | |
| Mehdi | Layers | | | |
| Mehdi | Models | | | |
| Douaâ | Neuron | [presentation](https://sylacefr.sharepoint.com/:p:/s/KPLR/EWgGzXjvPoRMlTNeiU-M8m8BTRh2AvfOpIoWV3TVeyXH6w?e=VrFcr3) | | |
| Douaâ | Neural Networks | | | |
| Douaâ | input layer | | | |
| Douaâ | hidden layer | | | |
| Douaâ | output layer | | | |
| Douaâ | weights | | | |
| Douaâ | polarisation node /bias | | | |
| Douaâ | learning rate | | | |
| Douaâ | activation function | | | [Trying Different activation functions](https://github.com/mehdi-lamrani/tensorflow/blob/main/clean/Trying_different_activation_functions_TF.ipynb)| 
| Douaâ | loss/cost function | | | [cost fnction in pure python + in tensorflow](https://colab.research.google.com/github/goodboychan/chans_jupyter/blob/main/_notebooks/2020-09-07-02-Cost-Minimization-using-Gradient-Descent.ipynb#scrollTo=BR4bzYcJy84g)| 
| Douaâ | back propagation | | | | [for math lovers](https://mmuratarat.github.io/2020-01-09/backpropagation) - [very simplified math](https://hmkcode.com/ai/backpropagation-step-by-step/)|
| Douaâ | gradient descent | | | | 
| Mehdi | **TensorBoard** | | | |

### DAY TWO
| Trainer Name | Topic |  Presentation | Exercice Notebook | Solution Notebook | External Ressource/example |
| ------ | ------ | ------ |  ------ | ------ | ------ |
|  | TF workshop series1 : linear regression |  |  | [linear regression with TF-solution](https://github.com/mehdi-lamrani/tensorflow/blob/main/clean/KPLR_TF_Linear_regression.ipynb) |  |
| Zineb  |  CNN : convolution  | [CNN presentation](https://docs.google.com/presentation/d/1R0aAw0dkHO0FMjWc7_dNb0vYR6lXM65OJxluRZcQ9X8/edit#slide=id.p) |  |  | | 
| Zineb  |  CNN : use cases  | [CNN presentation](https://docs.google.com/presentation/d/1R0aAw0dkHO0FMjWc7_dNb0vYR6lXM65OJxluRZcQ9X8/edit#slide=id.p) |  |  | | 
| Zineb  |  CNN : ReLu function | [CNN presentation](https://docs.google.com/presentation/d/1R0aAw0dkHO0FMjWc7_dNb0vYR6lXM65OJxluRZcQ9X8/edit#slide=id.p) |  | | | 
| Zineb  |  CNN : pooling types | [CNN presentation](https://docs.google.com/presentation/d/1R0aAw0dkHO0FMjWc7_dNb0vYR6lXM65OJxluRZcQ9X8/edit#slide=id.p) |  |  | | 
| Zineb  |  CNN : fully connected network | [CNN presentation](https://docs.google.com/presentation/d/1R0aAw0dkHO0FMjWc7_dNb0vYR6lXM65OJxluRZcQ9X8/edit#slide=id.p) |  | | | 
| Zineb  |  CNN : workshop  |  |  [Exercice_Convolutional_Neural_Networks_with_Tensorboard with Tensorboard](https://github.com/mehdi-lamrani/tensorflow/blob/main/clean/Exercice_Convolutional_Neural_Networks_with_Tensorboard.ipynb) | [Solution_Convolutional_Neural_Networks_with_Tensorboard with Tensorboard](https://github.com/mehdi-lamrani/tensorflow/blob/main/clean/Solution_Convolutional_Neural_Networks_with_Tensorboard.ipynb) | | 
| Zineb  |  Siamese Neural Network : workshop  | Cours+workshop | [Exercice_Face_Recognition_Siamese_network.ipynb](https://github.com/mehdi-lamrani/tensorflow/blob/main/clean/Exercice_Face_Recognition_Siamese_network.ipynb) | [Solution_Face_Recognition_Siamese_network.ipynb](https://github.com/mehdi-lamrani/tensorflow/blob/main/clean/Solution_Face_Recognition_Siamese_network.ipynb) | | 
| Douaâ | RNN : types and use cases | included in [presentation](https://sylacefr.sharepoint.com/:p:/s/KPLR/EWgGzXjvPoRMlTNeiU-M8m8BTRh2AvfOpIoWV3TVeyXH6w?e=VrFcr3) | | | |
| Douaâ | RNN : general architecture | included in [presentation](https://sylacefr.sharepoint.com/:p:/s/KPLR/EWgGzXjvPoRMlTNeiU-M8m8BTRh2AvfOpIoWV3TVeyXH6w?e=VrFcr3) | | | |
| Douaâ | RNN : forget gate | included in [presentation](https://sylacefr.sharepoint.com/:p:/s/KPLR/EWgGzXjvPoRMlTNeiU-M8m8BTRh2AvfOpIoWV3TVeyXH6w?e=VrFcr3) | | | |
| Douaâ | RNN : input gate | included in [presentation](https://sylacefr.sharepoint.com/:p:/s/KPLR/EWgGzXjvPoRMlTNeiU-M8m8BTRh2AvfOpIoWV3TVeyXH6w?e=VrFcr3) | | | |
| Douaâ | RNN : cell state | included in [presentation](https://sylacefr.sharepoint.com/:p:/s/KPLR/EWgGzXjvPoRMlTNeiU-M8m8BTRh2AvfOpIoWV3TVeyXH6w?e=VrFcr3) | | | |
| Douaâ | RNN : workshop |  | | [LSTM-Airline passengers forecasting with keras ](https://github.com/mehdi-lamrani/Neural-Networks/blob/main/LSTM_Forecast.ipynb)  -  [LSTM-Frozen Desserts Production with Tensorflow](https://github.com/doudi0101/ML-TPs/blob/main/RNN_Frozen_Desserts.ipynb) | to do : add markdown and comments from the video as hint, add 2021 data from kaggle | 

### DAY THREE

| Trainer Name | Topic |  Presentation | Exercice Notebook | Solution Notebook | External Ressource/example |
| ------ | ------ | ------ |  ------ | ------ | ----- |
|  |  |  |  |  |  |
| Douaâ | Reinforcement learning | [presentation](https://sylacefr.sharepoint.com/:p:/s/KPLR/EWgGzXjvPoRMlTNeiU-M8m8BTRh2AvfOpIoWV3TVeyXH6w?e=VrFcr3) |  | | Mehdi | Advanced TF Concepts | [presentation]() |  | | | 
| Mehdi | TF Parallel Distribution | [presentation]() |  | | | 
| Mehdi | Model Serving | [presentation]() |  | | | 


</details>

## Courses

<details>
  <summary>Details</summary>
  
### Stanford 

https://github.com/HaochunLiang/Stanford-CS20
https://github.com/HaochunLiang/Stanford-CS20/tree/master/Slides

https://www.youtube.com/watch?v=wG_nF1awSSY

### persson

https://github.com/aladdinpersson/Machine-Learning-Collection  
https://www.youtube.com/playlist?list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb

### heaton

https://github.com/mehdi-lamrani/deep-learning-modules  
https://www.youtube.com/playlist?list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN

### codebasics
https://github.com/codebasics/deep-learning-keras-tf-tutorial  
https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO

### Auto Diff 
(très important)

https://www.youtube.com/watch?v=boIOgsu-Q8E

### Distribution

https://www.youtube.com/watch?v=S1tN9a4Proc

</details>

## Tutos & resources : 

<details>
  <summary>Details</summary>
  
[Basic_Operations_Tensorflow](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb)

[Basic_linear_Regression_tensor](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb)

[Logistic Regression Example](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb)

[Build a recurrent neural network (LSTM) with TensorFlow](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/recurrent_network.ipynb)

[Convolutional Neural Network Example](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)

[Home_price_prediction](https://github.com/TahaSherif/Predicting-House-Prices-with-Regression-Tensorflow/blob/master/Predicting%20House%20Prices%20with%20Regression%20-%20Tensorflow%20.ipynb)

[Object_Detection](http://www.tensorflow.org/hub/tutorials/tf2_object_detection?hl=fr)

[Mnist_DATA](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb)

[Mnist_DATA_tuto](https://larevueia.fr/tensorflow/)

[Tensorflow_tuto](https://www.simplilearn.com/tutorials/deep-learning-tutorial/tensorflow)

</details>
