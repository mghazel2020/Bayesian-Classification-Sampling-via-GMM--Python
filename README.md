# Bayesian Classification and Sampling using Gaussian and Mixture Gaussian Models in Python

<img src="images/GMM - Bayes Classifier Sampling.png" width="1000"/>

## 1. Objectives

The objective of this project is to demonstrate how to model MINIST images using Gaussian Models (GM) and Mixture Gaussian Models (GMM)  and reconstruct each image from its GM or GMM models.

## 2.  Approach

In Bayes classification, we model p(x|y), rather than directly modeling p(y|x). This allows us to apply Generative Modeling to generate samples x for a each given class y. 

In this work, we consider two ways we can learn the posterior distribution p(x|y) from all the training data x that belongs to class y:

* Model the data as Gaussian Model (GM) distribution with mean and covariance estimated from the training data x belong to class y
* Model the data as Gaussian Mixture Model (GMM) distribution with mean and covariance estimated from the training data x belong to class y

Once we have modelled the training data x that belongs to class y, we can then generate sample from the estimated distribution to reconstruct examples of training MNIST images, x, belonging to different classes y.

We shall make use of the Scikit-learn GM and GMM distribution fitting and sampling and compare the performance if the two distribution fitting models discussed above. 


## 3. Data

We shall illustrate the PCA representation of the  MNIST database of handwritten digits, available from this page, which has a training set of 42,000 examples, and a test set of 18,000 examples. We shall illustrate sample images from this data sets in the next section.

## 4. Development

* Project: Bayesian Classifier Sampling of the MNIST Dataset based on Gaussian Models (GM) and Gaussian Mixture Models (GMM):
* The objective of this project is to demonstrate how to model MINIST images using Gaussian Models (GM) and Mixture Gaussian Models (GMM) and reconstruct each image from its GM or GMM models:
  * In this work, we consider two ways we can learn the posterior distribution p(x|y) from all the training data x that belongs to class y:
    * Model the data as Gaussian Model (GM) distribution with mean and covariance estimated from the training data x belong to class y
    * Model the data as Gaussian Mixture Model (GMM) distribution with multiple clusters means and covariances estimated from the training data x belong to class y.

* Author: Mohsen Ghazel (mghazel)
* Date: April 9th, 2021

### 4.1. Part 1: Python imports and global variables:

#### 4.1.1. Standard scientific Python imports:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt

<span style="color:#595979; "># import additional functionalities</span>
<span style="color:#200080; font-weight:bold; ">from</span> __future__ <span style="color:#200080; font-weight:bold; ">import</span> print_function<span style="color:#308080; ">,</span> division
<span style="color:#200080; font-weight:bold; ">from</span> builtins <span style="color:#200080; font-weight:bold; ">import</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">,</span> <span style="color:#400000; ">input</span>

<span style="color:#595979; "># import the multi-variate normal distribution </span>
<span style="color:#595979; "># sampling from Scipy</span>
<span style="color:#200080; font-weight:bold; ">from</span> scipy<span style="color:#308080; ">.</span>stats <span style="color:#200080; font-weight:bold; ">import</span> multivariate_normal <span style="color:#200080; font-weight:bold; ">as</span> mvn
<span style="color:#595979; "># import the Gaussian Mixure Model (GMM) distribution </span>
<span style="color:#595979; "># sampling from Scipy</span>
<span style="color:#200080; font-weight:bold; ">from</span> sklearn<span style="color:#308080; ">.</span>mixture <span style="color:#200080; font-weight:bold; ">import</span> BayesianGaussianMixture

<span style="color:#595979; "># import shuffle  from sklearn</span>
<span style="color:#200080; font-weight:bold; ">from</span> sklearn<span style="color:#308080; ">.</span>utils <span style="color:#200080; font-weight:bold; ">import</span> shuffle

<span style="color:#595979; "># import pandas</span>
<span style="color:#200080; font-weight:bold; ">import</span> pandas <span style="color:#200080; font-weight:bold; ">as</span> pd

<span style="color:#595979; "># random number generators values</span>
<span style="color:#595979; "># seed for reproducing the random number generation</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> seed
<span style="color:#595979; "># random integers: I(0,M)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> randint
<span style="color:#595979; "># random standard unform: U(0,1)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> random
<span style="color:#595979; "># time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime
<span style="color:#595979; "># I/O</span>
<span style="color:#200080; font-weight:bold; ">import</span> os
<span style="color:#595979; "># sys</span>
<span style="color:#200080; font-weight:bold; ">import</span> sys

<span style="color:#595979; "># display figure within the notebook</span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">3.4</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">8</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>


#### 4.1.2. Global variables:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># set the random_state seed = 100 for reproducibilty</span>
random_state_seed <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span>

<span style="color:#595979; "># the number of visualized images</span>
num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span>
</pre>

### 4.2. Part 2: Load MNIST Dataset:

* We use the MINIST dataset, which was downloaded from the following link:

  * Kaggle: Digit Recognizer: https://www.kaggle.com/c/digit-recognizer/data
  * The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.
  * Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
  * The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
  * Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero)
  
#### 4.2.1. Load and normalize the training data set:



<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># read the training data set</span>
data <span style="color:#308080; ">=</span> pd<span style="color:#308080; ">.</span>read_csv<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'../large_files/train.csv'</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>values<span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>float32<span style="color:#308080; ">)</span>
<span style="color:#595979; "># normalize the training data to [0,1]:</span>
x_train <span style="color:#308080; ">=</span> data<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span> <span style="color:#44aadd; ">/</span> <span style="color:#008c00; ">255</span>
<span style="color:#595979; "># format the class type to integer</span>
y_train <span style="color:#308080; ">=</span> data<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>int32<span style="color:#308080; ">)</span>
<span style="color:#595979; "># shuffle the data</span>
x_train<span style="color:#308080; ">,</span> y_train <span style="color:#308080; ">=</span> shuffle<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Display a summary of the training data:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># the number of training images</span>
num_train_images <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Training data:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"x_train.shape: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"y_train.shape: "</span><span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of training images: "</span><span style="color:#308080; ">,</span> num_train_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Image size: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Classes/labels:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">42000</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">784</span><span style="color:#308080; ">)</span>
y_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">42000</span><span style="color:#308080; ">,</span><span style="color:#308080; ">)</span>
Number of training images<span style="color:#308080; ">:</span>  <span style="color:#008c00; ">42000</span>
Image size<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">784</span><span style="color:#308080; ">,</span><span style="color:#308080; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span> <span style="color:#008c00; ">2</span> <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">4</span> <span style="color:#008c00; ">5</span> <span style="color:#008c00; ">6</span> <span style="color:#008c00; ">7</span> <span style="color:#008c00; ">8</span> <span style="color:#008c00; ">9</span><span style="color:#308080; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.2. Visualize some of the training images and their associated targets:

##### 4.2.2.1. First implement a visualization functionality to visualize the number of randomly selected images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">"""</span>
<span style="color:#595979; "># A utility function to visualize multiple images:</span>
<span style="color:#595979; ">"""</span>
<span style="color:#200080; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#308080; ">,</span> dataset_flag <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; ">"""To visualize images.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#595979; ">&nbsp;&nbsp;"""</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  <span style="color:#595979; "># the suplot grid shape:</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  num_rows <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#595979; "># the number of columns</span>
  num_cols <span style="color:#308080; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#595979; "># setup the subplots axes</span>
  fig<span style="color:#308080; ">,</span> axes <span style="color:#308080; ">=</span> plt<span style="color:#308080; ">.</span>subplots<span style="color:#308080; ">(</span>nrows<span style="color:#308080; ">=</span>num_rows<span style="color:#308080; ">,</span> ncols<span style="color:#308080; ">=</span>num_cols<span style="color:#308080; ">,</span> figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
  <span style="color:#595979; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#308080; ">(</span>random_state_seed<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># iterate over the sub-plots</span>
  <span style="color:#200080; font-weight:bold; ">for</span> row <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_rows<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      <span style="color:#200080; font-weight:bold; ">for</span> col <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_cols<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># get the next figure axis</span>
        ax <span style="color:#308080; ">=</span> axes<span style="color:#308080; ">[</span>row<span style="color:#308080; ">,</span> col<span style="color:#308080; ">]</span><span style="color:#308080; ">;</span>
        <span style="color:#595979; "># turn-off subplot axis</span>
        ax<span style="color:#308080; ">.</span>set_axis_off<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>num_train_images<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the training image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> y_train<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># dataset_flag = 2: Test data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>num_test_images<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the test image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>x_test<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> y_test<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># display the image</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        ax<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>image<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span>plt<span style="color:#308080; ">.</span>cm<span style="color:#308080; ">.</span>gray_r<span style="color:#308080; ">,</span> interpolation<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'nearest'</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># set the title showing the image label</span>
        ax<span style="color:#308080; ">.</span>set_title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'y ='</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>label<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> size <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#308080; ">)</span>
</pre>

##### 4.2.2.2. Call the function to visualize the randomly selected training images:

<img src="images/sample-train-images.png" width="1000"/>

##### 4.2.2.3. Examine the number of images for each class of the training and testing subsets:

<img src="images/train-images-histogram.png" width="1000"/>

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># create a histogram of the number of images in each class/digit:</span>
<span style="color:#200080; font-weight:bold; ">def</span> plot_bar<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">,</span> relative<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    width <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#200080; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#200080; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#595979; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#308080; ">,</span> counts <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> return_counts<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
    sorted_index <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>argsort<span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span>
    unique <span style="color:#308080; ">=</span> unique<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
     
    <span style="color:#200080; font-weight:bold; ">if</span> relative<span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot as a percentage</span>
        counts <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y<span style="color:#308080; ">)</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'% count'</span>
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot counts</span>
        counts <span style="color:#308080; ">=</span> counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'count'</span>
         
    xtemp <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>arange<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>bar<span style="color:#308080; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#308080; ">,</span> counts<span style="color:#308080; ">,</span> align<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'center'</span><span style="color:#308080; ">,</span> alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">.7</span><span style="color:#308080; ">,</span> width<span style="color:#308080; ">=</span>width<span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span>xtemp<span style="color:#308080; ">,</span> unique<span style="color:#308080; ">,</span> rotation<span style="color:#308080; ">=</span><span style="color:#008c00; ">45</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'digit'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span>ylabel_text<span style="color:#308080; ">)</span>
 
plt<span style="color:#308080; ">.</span>suptitle<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Frequency of images per digit'</span><span style="color:#308080; ">)</span>
plot_bar<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>
    <span style="color:#1060b6; ">'train ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
</pre>

<img src="images/train-images-histogram.png" width="1000"/>


### 4.3. Part 3: Implement the Bayesian Classification and Sampling using Gaussian Models (GM):

#### 4.3.1. Implement a utility function

* This function that the data values ranges between 0 and 1:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> clamp_sample<span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;This function that the data values ranges between 0 and 1</span>
<span style="color:#595979; ">&nbsp;&nbsp;"""</span>
  x <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>minimum<span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
  x <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>maximum<span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
  <span style="color:#200080; font-weight:bold; ">return</span> x
</pre>

#### 4.3.2. Implement the Bayesian Classifier Classifier based on Gaussian Models (GM):

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">class</span> BayesClassifierGM<span style="color:#308080; ">:</span>
  <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;This implements the GM-based BayesClassifier class</span>
<span style="color:#595979; ">&nbsp;&nbsp;"""</span>
  <span style="color:#200080; font-weight:bold; ">def</span> fit<span style="color:#308080; ">(</span>self<span style="color:#308080; ">,</span> X<span style="color:#308080; ">,</span> Y<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; "># assume classes are numbered 0...K-1</span>
    self<span style="color:#308080; ">.</span>K <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span><span style="color:#400000; ">set</span><span style="color:#308080; ">(</span>Y<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># Initialize the gaussians list to empty</span>
    self<span style="color:#308080; ">.</span>gaussians <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>
    <span style="color:#595979; "># Initialize the prior probabilities</span>
    self<span style="color:#308080; ">.</span>p_y <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>K<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># iterate ove rthe classes</span>
    <span style="color:#200080; font-weight:bold; ">for</span> k <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>K<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      <span style="color:#595979; "># get the training data for class k</span>
      Xk <span style="color:#308080; ">=</span> X<span style="color:#308080; ">[</span>Y <span style="color:#44aadd; ">==</span> k<span style="color:#308080; ">]</span>
      <span style="color:#595979; "># keep track of the number of </span>
      <span style="color:#595979; "># training examples in this class</span>
      self<span style="color:#308080; ">.</span>p_y<span style="color:#308080; ">[</span>k<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>Xk<span style="color:#308080; ">)</span>
      <span style="color:#595979; "># compute the mean of the training data in this class</span>
      mean <span style="color:#308080; ">=</span> Xk<span style="color:#308080; ">.</span>mean<span style="color:#308080; ">(</span>axis<span style="color:#308080; ">=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
      <span style="color:#595979; "># compute the covariance of the training data in this class</span>
      cov <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>cov<span style="color:#308080; ">(</span>Xk<span style="color:#308080; ">.</span>T<span style="color:#308080; ">)</span>
      <span style="color:#595979; "># store the means and covariance in a dictionary</span>
      g <span style="color:#308080; ">=</span> <span style="color:#406080; ">{</span><span style="color:#1060b6; ">'m'</span><span style="color:#308080; ">:</span> mean<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">'c'</span><span style="color:#308080; ">:</span> cov<span style="color:#406080; ">}</span>
      <span style="color:#595979; "># append g to the guassians list</span>
      self<span style="color:#308080; ">.</span>gaussians<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>g<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># normalize p(y) to convert frequencies to probabilities</span>
    self<span style="color:#308080; ">.</span>p_y <span style="color:#44aadd; ">/</span><span style="color:#308080; ">=</span> self<span style="color:#308080; ">.</span>p_y<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
  
  <span style="color:#200080; font-weight:bold; ">def</span> sample_given_y<span style="color:#308080; ">(</span>self<span style="color:#308080; ">,</span> y<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;This function generates a gaussian random </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;sample coresponding to class y.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#595979; "># get the Guassian statistics (mean and covariance) for class y</span>
    g <span style="color:#308080; ">=</span> self<span style="color:#308080; ">.</span>gaussians<span style="color:#308080; ">[</span>y<span style="color:#308080; ">]</span>
    <span style="color:#595979; "># generate a multi-variate Guassian random sample with the </span>
    <span style="color:#595979; "># corresponding statistics</span>
    <span style="color:#595979; "># - ensure that the random samples are between 0 and 1</span>
    <span style="color:#595979; "># - return the generated sample</span>
    <span style="color:#200080; font-weight:bold; ">return</span> clamp_sample<span style="color:#308080; ">(</span> mvn<span style="color:#308080; ">.</span>rvs<span style="color:#308080; ">(</span>mean<span style="color:#308080; ">=</span>g<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'m'</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> cov<span style="color:#308080; ">=</span>g<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'c'</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#308080; ">)</span>

  <span style="color:#200080; font-weight:bold; ">def</span> sample<span style="color:#308080; ">(</span>self<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;This function generates a gaussian random </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;sample coresponding to a randomly selected class y</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    y <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>choice<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>K<span style="color:#308080; ">,</span> p<span style="color:#308080; ">=</span>self<span style="color:#308080; ">.</span>p_y<span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">return</span> y<span style="color:#308080; ">,</span> clamp_sample<span style="color:#308080; ">(</span> self<span style="color:#308080; ">.</span>sample_given_y<span style="color:#308080; ">(</span>y<span style="color:#308080; ">)</span> <span style="color:#308080; ">)</span>
</pre>

#### 4.3.3. Implement the GM-based Bayesian Classifier fitting and sampling:

##### 4.3.3.1. Instantiate and fit the GM-based Bayesian Classifier:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Instantiate the Gaussian Model (GM) based Bayesian Classifer</span>
clf <span style="color:#308080; ">=</span> BayesClassifierGM<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># Fit the Bayesian Classifier to the training data</span>
clf<span style="color:#308080; ">.</span>fit<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">)</span>
</pre>

##### 4.3.3.2. Use the GM-based Bayesian Classifier to generate samples from each class:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Iterarate over the classes</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#200080; font-weight:bold; ">for</span> k <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>clf<span style="color:#308080; ">.</span>K<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># Step 1: generate Gaussian sample</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    sample <span style="color:#308080; ">=</span> clf<span style="color:#308080; ">.</span>sample_given_y<span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># generate the mean</span>
    mean <span style="color:#308080; ">=</span> clf<span style="color:#308080; ">.</span>gaussians<span style="color:#308080; ">[</span>k<span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#1060b6; ">'m'</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># Step 2: generate sample mean</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># create a figure and set its axis</span>
    fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># create the figure </span>
    plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># display the sample</span>
    plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>sample<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"GM-based Bayes Classifier - Generated sample for class: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>mean<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Mean of all training data for class: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<table>
  <tr>
    <td> <img src="images/GM-Bayes-Classifier-Sampling-All-Classes-0-4.png" width="1000"></td>
  </tr>
  <tr>
    <td> <img src="images/GM-Bayes-Classifier-Sampling-All-Classes-5-9.png" width="1000"></td>
  </tr>
</table>

##### 4.3.2.3. Generate a random sample from a random class:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># generate a random sample froma random class</span>
k<span style="color:#308080; ">,</span> sample <span style="color:#308080; ">=</span> clf<span style="color:#308080; ">.</span>sample<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># reshape the sample into 28x28 image</span>
sample <span style="color:#308080; ">=</span> sample<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>sample<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Random Sample from a random selected class: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/GM-Bayes-Classifier-Sampling-Random-Class-3.png" width="500"/>

### 4.4. Part 4: Implement the Bayesian Classification and Sampling based on Gaussian Mixture Models (GMM):

#### 4.4.1. Implement the Bayesian Classifier Classifier based on Gaussian Mixture Models (GMM):


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">class</span> BayesClassifierGMM<span style="color:#308080; ">:</span>
  <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;This implements the GMM-based BayesClassifier class</span>
<span style="color:#595979; ">&nbsp;&nbsp;"""</span>
  <span style="color:#200080; font-weight:bold; ">def</span> fit<span style="color:#308080; ">(</span>self<span style="color:#308080; ">,</span> X<span style="color:#308080; ">,</span> Y<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; "># assume classes are numbered 0...K-1</span>
    self<span style="color:#308080; ">.</span>K <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span><span style="color:#400000; ">set</span><span style="color:#308080; ">(</span>Y<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># initialize the gaussian list to empty</span>
    self<span style="color:#308080; ">.</span>gaussians <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>
    <span style="color:#595979; "># initialize the prior probabilities to zeros</span>
    self<span style="color:#308080; ">.</span>p_y <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>K<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># iterate ove rthe classes</span>
    <span style="color:#200080; font-weight:bold; ">for</span> k <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>K<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      <span style="color:#595979; "># get the training data corresponding to class k</span>
      Xk <span style="color:#308080; ">=</span> X<span style="color:#308080; ">[</span>Y <span style="color:#44aadd; ">==</span> k<span style="color:#308080; ">]</span>
      <span style="color:#595979; "># update the probability to the number of training images in class: k</span>
      self<span style="color:#308080; ">.</span>p_y<span style="color:#308080; ">[</span>k<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>Xk<span style="color:#308080; ">)</span>
      <span style="color:#595979; "># instantiate a GMM model with max number of clusters = 10 </span>
      gmm <span style="color:#308080; ">=</span> BayesianGaussianMixture<span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span>
      <span style="color:#595979; "># fit the GMM model to the training data corrsponding to class: k</span>
      gmm<span style="color:#308080; ">.</span>fit<span style="color:#308080; ">(</span>Xk<span style="color:#308080; ">)</span>
      <span style="color:#595979; "># append the generated GMM model to Gaussian</span>
      self<span style="color:#308080; ">.</span>gaussians<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>gmm<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># normalize p(y) to convert frequency to probability</span>
    self<span style="color:#308080; ">.</span>p_y <span style="color:#44aadd; ">/</span><span style="color:#308080; ">=</span> self<span style="color:#308080; ">.</span>p_y<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

  <span style="color:#200080; font-weight:bold; ">def</span> sample_given_y<span style="color:#308080; ">(</span>self<span style="color:#308080; ">,</span> y<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;This function generates samples from class y</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#595979; "># get the GMM model corresponding to class y</span>
    gmm <span style="color:#308080; ">=</span> self<span style="color:#308080; ">.</span>gaussians<span style="color:#308080; ">[</span>y<span style="color:#308080; ">]</span>
    <span style="color:#595979; "># generate a sample using the GMM model corresponding to class y</span>
    sample <span style="color:#308080; ">=</span> gmm<span style="color:#308080; ">.</span>sample<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; ">#---------------------------------------------------</span>
    <span style="color:#595979; "># Note:</span>
    <span style="color:#595979; ">#---------------------------------------------------</span>
    <span style="color:#595979; "># - sample returns a tuple containing 2 things:</span>
    <span style="color:#595979; "># 1) the sample</span>
    <span style="color:#595979; "># 2) which cluster it came from</span>
    <span style="color:#595979; ">#---------------------------------------------------</span>
    <span style="color:#595979; "># we'll use (2) to obtain the means so we can plot</span>
    <span style="color:#595979; "># them like we did in the previous script</span>
    <span style="color:#595979; "># we cheat by looking at "non-public" params in</span>
    <span style="color:#595979; "># the sklearn source code</span>
    <span style="color:#595979; ">#---------------------------------------------------</span>
    <span style="color:#595979; "># the mean of the cluster from which the sample was generated</span>
    mean <span style="color:#308080; ">=</span> gmm<span style="color:#308080; ">.</span>means_<span style="color:#308080; ">[</span>sample<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">]</span>
    <span style="color:#595979; "># return teh generated sample and the cluster mean</span>
    <span style="color:#200080; font-weight:bold; ">return</span> clamp_sample<span style="color:#308080; ">(</span> sample<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> mean<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>

  <span style="color:#200080; font-weight:bold; ">def</span> sample<span style="color:#308080; ">(</span>self<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;This function generates a gaussian random </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;sample coresponding to a randomly selected class y</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#595979; "># randomly generate the class</span>
    y <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>choice<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>K<span style="color:#308080; ">,</span> p<span style="color:#308080; ">=</span>self<span style="color:#308080; ">.</span>p_y<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># randomly generate the class</span>
    <span style="color:#200080; font-weight:bold; ">return</span> y<span style="color:#308080; ">,</span> clamp_sample<span style="color:#308080; ">(</span> self<span style="color:#308080; ">.</span>sample_given_y<span style="color:#308080; ">(</span>y<span style="color:#308080; ">)</span> <span style="color:#308080; ">)</span>
</pre>


#### 4.4.2. Implement the GMM-based Bayesian Classifier fitting and sampling:

##### 4.4.2.1. Instantiate and fit the GMM-based Bayesian Classifier:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Instantiate the Gaussian Mixture Model (GMM) based Bayesian Classifier</span>
clf <span style="color:#308080; ">=</span> BayesClassifierGMM<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># Fit the Bayesian Classifier to the training data</span>
clf<span style="color:#308080; ">.</span>fit<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">)</span>
</pre>


##### 4.4.2.2. Use the GMM-based Bayesian Classifier to generate samples from each class:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Iterate over the classes</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#200080; font-weight:bold; ">for</span> k <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>clf<span style="color:#308080; ">.</span>K<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># Step 1: show one sample for each class</span>
    <span style="color:#595979; ">#  - also show the mean image learned</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    sample<span style="color:#308080; ">,</span> mean <span style="color:#308080; ">=</span> clf<span style="color:#308080; ">.</span>sample_given_y<span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span>

    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># Step 2: generate sample mean</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># create a figure and set its axis</span>
    fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># create the figure </span>
    plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># display the sample</span>
    plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>sample<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"GMM-based Bayes Classifier - Generated sample for class: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>mean<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Sampled Cluster Mean for class: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<table>
  <tr>
    <td> <img src="images/GMM-Bayes-Classifier-Sampling-All-Classes-0-4.png" width="1000"></td>
  </tr>
  <tr>
    <td> <img src="images/GMM-Bayes-Classifier-Sampling-All-Classes-5-9.png" width="1000"></td>
  </tr>
</table>

##### 4.4.2.3. Generate a random sample from a random class:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># generate a random sample from a random class</span>
k<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>sample<span style="color:#308080; ">,</span> mean<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> clf<span style="color:#308080; ">.</span>sample<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>sample<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Random Sample from a random selected class: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>


<img src="images/GMM-Bayes-Classifier-Sampling-Random-Class-7.png" width="500"/>




<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">10</span> <span style="color:#008c00; ">03</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">29</span><span style="color:#308080; ">:</span><span style="color:#008000; ">48.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>

## 5. Analysis

* In view of the presented results, we make the following observations:

  * We implemented two ways we can learn the posterior distribution p(x|y) from all the training data x that belongs to class y:
  * Model the data as Gaussian Model (GM) distribution with mean and covariance estimated from the training data x belong to class y:
  * The MNIST samples generated from the estimated Gaussian Models (GM)-based Bayes Classifier are readable but suffer from significant blurring artifacts.
  * Model the data as Gaussian Mixture Model (GMM) distribution with multiple clusters means and covariances estimated from the training data x belong to class y:
  * The MNIST samples generated from the estimated Gaussian Mixture Models (GMM)-based Bayes Classifier are much better but still suffer from some degree of blurring artifacts.

## 6. Future Work

* We plan to explore the following related issues:

  * To explore implementing Auto-Encoders (VA), Variational Auto-Encoders (V-AE) and Generative Adversarial Networks (GANs) models to model and reconstruct MINIST images from noise.
  * These more advanced models are expected to generated reconstructed images with much better quality.

## 7. References

1. Kaggle. Digit Recognizer: Learn computer vision fundamentals with the famous MNIST data. https://www.kaggle.com/c/digit-recognizer/data
2.. Yann LeCun et. al. THE MNIST DATABASE of handwritten digits. http://yann.lecun.com/exdb/mnist/ 
3. Python Data Science Handbook. In Depth: Gaussian Mixture Models. https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html 
4. Chris Fonnesbeck. Fitting Gaussian Process Models in Python. https://blog.dominodatalab.com/fitting-gaussian-process-models-python/ 
5. Jonathan DEKHTIARA. Guide Through Generative Models - Part 1: How To Generate new Data with Bayesian Sampling. https://www.born2data.com/2017/generative_models-part1.html
 
 
