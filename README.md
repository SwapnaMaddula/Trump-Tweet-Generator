# Trump-Tweet-Generator
<b>INDIVIDUAL-PROJECT REPORT<b>
DONALD TRUMP TWEET GENERATOR
MOTIVATION
The motivation for this project was to explore natural language processing and text generation techniques. The goal of this project is to impersonate a human’s character utilizing tools of machine learning. This is a way to showcase machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human. I thought that a pertinent application of NLP, given the current news environment, was to analyze President Donald Trump’s tweets. From his tweets, I was successful in clustering his tweets into distinct groups, and generate original tweets in his linguistic style.
PROBLEM STATEMENT
The problem can be stated as “Given a sample tweet or type of tweet, a new tweet which looks like the one Donald Trump may write is being generated.”
APPROACH
In the process of creating a tweet generator, I created a one-to-sequence model that can generate tweets like Trump. The steps to create the tweet generator as follows: 
•	Collection and inspection of data i.e. Trump Tweets.
•	Preprocessing or curation of dataset.
•	Create words vectors using pre-trained word vectors (GloVe — Twitter 200d).
•	Use PCA to reduce the dimensionality.
•	Cluster the tweets with K-Means Clustering.
•	Order the tweets by their PCA values.
•	Limit the tweets by the quality of their text.
•	Build and train the model
•	Generation of new tweets
In the end, we can use such a simple input i.e. one value, to generate a tweet. In addition to this, from the way input data is created one can even control the style of our generated tweet.
DATA PROCESSING
Data Collection and Processing
The first and foremost step in this project is dataset collection. I am using reliable data set from kaggle (https://www.kaggle.com/kingburrito666/better-donald-trump-tweets) which contains 7375 tweets made by Donald Trump. 
Next step is curating data. In this pre-processing step, the unwanted characters and URL's are removed from each tweet. The formatting of the text is performed in this step to create unwanted character free tweets to be used as training data. Thus, the tweets are preprocessed to remove all the unnecessary characters and spacing’s which does not add any more information. Once all the links and unhelpful characters are removed, next step is to reformat the text so that we can maximize the use of GloVe’s pre-trained word vectors. 
Word Vector Generation
In this case to generate word vectors, we use “Global Vectors for word representation” (GloVe). https://nlp.stanford.edu/projects/glove/  provides curated data that contains 1,193,514 word embedding’s, each with 200 dimensions. 
To make word vectors, one should create an embedding index from GloVe’s word vectors. We start using this embedding index to create an ‘average’ embedding for each tweet. We start with an ‘empty’ embedding (all 200 values [for the 200 dimensions] will be 0). If a word from a tweet is in the index, its embedding will be added to the ‘empty’ embedding. If a word from the tweet is not in the index, nothing will be added to the ‘empty’ embedding. 
After each word from the tweet has been accounted for, we average each dimension by the number of words in the tweet. These averaged dimensions will create our ‘average’ embedding for each tweet. To create these tweet embedding’s, there are other methods, such as Word2Vec. The benefit of Word2Vec is that there wouldn’t be any null embedding’s, but given the amount of data that we are working with, only 7375 tweets, it would be better to make use of GloVe’s larger dataset. 
Thus, formatting of the tweets is performed in this step to create fewer nulls embedding in between the words.
Dimensionality Reduction
Our next step is to reduce the dimensionality of our data from 200 to 1. Number of dimensions of these vectors are reduced using “Principle Component Analysis (PCA)” Algorithm. Using Principal Component analysis, the dimensionality is reduced from 200 which derived in earlier step to 1 dimension. We set the reduction to 1 to simplify and organize our input data, as well as make it easy to generate the type of tweet that we want, eg. if we want a tweet with more hashtags or just text.
Now that each of our tweets have a PCA value, we are going to rank them from the smallest value to the largest, so that similar tweets will be closer together. We multiply our pca_labels by a factor two to make it simpler to generate new tweets. Although we will not be using all tweets to train our model, all training tweets will have even numbers. To ensure that you are generating a new tweet, all you need to do is use an odd number as your input. This generated tweet should look like tweets in the training data that have values near your input value.
K-Means Clustering
	The entire PCA modified tweets are grouped into 4 clusters. Training data is prepared by making few manipulations on this data like sorting the data by length etc. To understand the types of tweets that Trump makes, we are going to use K-Means to divide the tweets into groups. pca_tweets will be our input for K-Means, and after checking the results of this function using 3-10 clusters, 4 clusters stood as best number for the Trump’s tweets to be made into distinct groups.
Each group contains the following number of tweets: 1 (Red): 315, 2 (Orange): 2600, 3 (Yellow): 1674, 4 (Blue): 2782. The tweets are coloured by their KMeans labels, but the input data to TSNE was the embed_tweets. 

 
We can take a look at how well the PCA labels compare with the tweet groups. Given that the tweet groups were created using PCA data, we can see four clear groups. This also provides a nice visualization to compare the different sizes of the tweet groups, and can help you to select a tweet type when you want to generate a tweet. In the figure below, are some tweets from each group.

 
SOLUTION IMPLEMENTATION
Using tensor flow, RNN (Recurrent Neural Network) is used for model generation. It is a multi-layered model which uses deep learning techniques for effective tweet generation. Using encoding and decoding layers on training and prediction data, appropriate model is generated. Once the model is generated, epochs are repeated in training until the training loss is not changed.
Recurrent Neural Networks (RNN)
The idea behind RNNs is to make use of sequential information. In a traditional neural network, we assume that all inputs (and outputs) are independent of each other. But for many tasks that’s a very bad idea. If you want to predict the next word in a sentence you better know which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Another way to think about RNNs is that they have a “memory” which captures information about what has been calculated so far. In theory RNNs can make use of information in arbitrarily long sequences, but in practice they are limited to looking back only a few steps (more on this later). Here is what a typical RNN looks like:
 
RNN’s have many applications like language translation, text summarization, text generation, etc.
Model Generation and Training
Below mentioned are the steps for model generation and training:
•	First step, create placeholders for our model’s inputs. In the code which is written to generate RNN model, learning rate and keep probability do not have a shape parameter. This is because the default shape is none and is taken the same.
•	As part of processing encoding layer inputs, tensor flow library is used. The function tf.strided_slice() will remove the final word from each batch. Appended to the start of each batch will be the token <GO>. This formatting is necessary for creating the embedding’s for our decoding layer.
•	Next is an encoding layer to encode data. LSTM cells typically outperform GRU cells for seq2seq tasks, such as this one. Making the encoder bidirectional proved to be much more effective than a simple feed forward network. We return only the encoder’s state because it is the input for our decoding layer. Simply put, the weights of the encoding cells are what interest us.
•	We have two different functions, one is decoder which is used for training data and the other, decoder which is used for prediction. Predictions are based on the outputs from the training data. 
•	We use a decoding RNN cell, and a fully connected layer to create our training and inference logits. We use tf.variable_scope() to reuse the variables from training data.
•	Finally, we tie everything together and generate the outputs for our model. Like initializing weights and biases, embedding’s are also initialized to their best value. 
Set the hyper parameters to generate good results, minimizing the loss function in fewer epochs. A larger network could produce better results, but given the number of iterations it is better to do it with a GPU. Using learning rate decay is always better, Adam optimizer is used in our case. As your model tries to find the optimal weights, it needs to update these values with smaller increments, so a shrinking learning rate is beneficial.
Now this model is trained with the training data for epochs until, the loss function stays stable for 3 iterations i.e. it converges. Once this happens, the training is stopped. This RNN model is used for making predictions i.e. generating tweets.

Training Data Preparation
To help the model generate the best tweets, we are going limit our training data with a few measures. Training data is the most important part of a model. Without good training data, a model will be unable to generate good outputs.
The first thing do is to build a vocabulary to integer (vocab_to_int) dictionary and int_to_vocab dictionary, which include only words that appear 10 or more times in Trump’s tweets. It will help our generated tweets to sound more like Trump’s typical tweets and help the model to better understand what each word means because it will see it at least 10 times. If the threshold was just 2 or 3, the model would struggle to understand and use words that rarely appear.
Next, we limit the lengths of the tweets that we will use. 25 words is selected as the maximum length for a tweet in the training data. This value was chosen because it was the maximum length that could still be learned rather well by the model. Any longer and the model struggles to learn the tweet, any shorter and we would be further limiting our training data. We are also going to set an <UNK> limit. <UNK> is a token used to represent words that are not included in our training vocabulary. If more than one <UNK> token is present in a tweet, the tweet will not be used to train the model.  
As a final step to prepare the data for the model, we sort the tweets by length. Doing so will help the model to train faster because the earlier batches will contain shorter tweets. 
New Tweets Generation
There are two different methods to generate tweets. They are as follows:
Method 1: Find a Similar Tweet
With this method, one type words or a phrase that will be matched to the most similar tweet.
•	In input variable, write any text that you want to be matched.
•	This tweet will be cleaned then converted to a vector using the same method as our input data when we made the ‘average’ embedding’s.
•	The tweet’s ‘average’ embedding will be used to find its PCA value. This value will be matched to the tweet with the closest value.
•	The matched tweet’s PCA value will be used to find its PCA label. This label is the input to the model.
The PCA label of the input tweet will be matched to the closest label that was used to train the model. The text that is related to the closest label will be printed out. The content of input and output tweets may be rather different, but their structure remains similar.
Method 2: Input a value
This is this the simpler option, and includes a bit more randomness. All one needs to do is select a tweet type, by referring the tweet groups’ ranges to control for the style of tweet. One can also control the length of the tweet that you want to generate by setting the sequence length’s value, but it set to a random integer in this case.
IMPACT
This project is a close step towards impersonating a person, based on his actions, in this case his tweets. A person tweets can be solely produced by a machine with zero involvement of the person. This a huge improvement since the person need not make the actual tweet and is auto generated by the machine. 
CONCLUSION
	A tweet generator is developed which can mimic Donald Trump and generate tweets which look similar to his tweets. Future works in this area of text generation are possible. The tweet generator can be made so perfect in a way that one cannot differentiate if the tweet is coming from the actual person or the machine. This can happen with better training of the tweet generator.
REFERENCES
Zhang, Wei-Nan, et al. "Neural personalized response generation as domain adaptation." World Wide Web (2017): 1-20.
