 
 
<h1>Stock market prediction using LSTM  </h1>  


>  Python deep learning model with Keras Long Short-Term Memory (LSTM) to predict the future behavior of Petrobras stock prices.


The purpose of this project was to get started forecasting time series with LSTM models. Based on the historical daily prices of Petrobras stocks from 2012 to 2018, the model predicts the opening prices of 2019. I still have to create different versions and improve the results. 


#### Currently in progress ! 💻 

<!-- /TOC -->
<h1>Table of Contents</h1>
 
- [1. Overview](#1-overview)
- [2. Quick Start](#2-quick-start)
- [3. What is LSTM and how it works](#3-what-is-lstm-and-how-it-works)
- [4. Dataset](#4-dataset)
- [5. Approach](#5-approach)
- [6. Dependencies](#6-dependencies)
- [7. Results](#7-results)
- [8. Useful](#8-useful)

<!-- /TOC -->

## 1. Overview 

In this project I use Keras Long Short-Term Memory (LSTM) Model to Predict Petrobras Stock Prices. LSTMs are very powerful in sequence prediction problems because they're able to store past information. This is important here because the previous price of a stock is crucial in predicting its future price. Stock market data is a great choice for this kind of study because it’s quite regular and widely available to everyone.	


## 2. Quick Start  

[Checkout](https://nbviewer.jupyter.org/github/alicevillar/stock_market_prediction_lstm/blob/main/Untitled.ipynb) a static version of the notebook with Jupyter NBViewer from the comfort of your web browser.

## 3. What is LSTM and how it works
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. LSTMs have feedback connections, which enables it to process entire sequences of data (e.g. time series). Rather than treating each point in the sequence independently, LSTMs retains useful information about previous data in the sequence to help with the processing of new data points. As a result, LSTMs are particularly good at processing sequences of data such as text, speech and general time-series.

## 4. Dataset  
For this project I used the [Yahoo Finance]( https://finance.yahoo.com/quote/PBR?p=PBR&.tsrc=fin-srch) for the historical daily prices of Petrobras stocks.
- [Training dataset](https://github.com/alicevillar/ltsm_petrobras/blob/main/dataset_2012_2018.csv): historical daily prices of Petrobras stocks from 2012 to 2018
- [Test dataset](https://github.com/alicevillar/ltsm_petrobras/blob/main/dataset_2019.csv): historical daily prices of Petrobras stocks of 2019

## 5. Approach

* PART 1: Data Handling -> Importing Data with Pandas, Cleaning Data, Data description.
* PART 2: Data Analysis -> Supervised ML Technique: LSTM algorithm 
* PART 3: Valuation of the Analysis -> Plotting results

## 6. Dependencies  
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [sciKit-learn](https://scikit-learn.org/)
* [matplotlib](https://matplotlib.org/)
* [tf.keras]( https://www.tensorflow.org/guide/keras?hl=pt-br)

## 7. Results  

![print](petrobras_plot.JPG)


## 8. Useful  

* [kd Nuggets - Using a Keras Long Short-Term Memory (LSTM) Model to Predict Stock Prices](https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html)
* [Towards Datas Cience - LSTM Networks | A Detailed Explanation](https://towardsdatascience.com/lstm-networks-a-detailed-explanation-8fae6aefc7f9)
* [PUC Goiais - Seleção de portfólio de ações por meio de otimização metaheurística híbrida com redes neurais LSTM](http://tede2.pucgoias.edu.br:8080/bitstream/tede/4512/2/Danilo%20Alc%c3%a2ntara%20Milhomem.pdf)
* [Medium - Deep Learning and Portfolio Management](https://medium.com/geekculture/deep-learning-and-portfolio-management-b3b983528a06)
* [Coursera - Investment Management with Python and Machine Learning Specialization](https://www.coursera.org/specializations/investment-management-python-machine-learning)

