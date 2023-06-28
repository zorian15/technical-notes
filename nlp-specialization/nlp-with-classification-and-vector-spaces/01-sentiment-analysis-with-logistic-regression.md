# Sentiment analysis with logistic regression

In this [module](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/1), I will learn how to extract features from text into numerical vectors, then build a binary classifier for tweets using a logistic regression model.

## Learning Objectives

- Sentiment analysis
- Logistic regression
- Data pre-processing
- Calculating word frequencies
- Feature extraction
- Vocabulary creation
- Supervised learning

## Lecture Notes

### Supervised ML & Sentiment Analysis

![Supervised ML workflow](figures/supervised-ml.png)
- We have input features $X$ and output labels $Y$
- We train a prediction function to predict labels $\hat{Y}$
- Our cost function $Cost(Y, \hat{Y})$, compares the actual output labels $Y$ to our predicted labels $\hat{Y}$
- We then use our cost function to update the parameters $\theta$ in our prediction function
  - We want to find $\theta$ values that minimize the cost function

![Sentiment analysis overview](figures/sentiment-analysis.png)
- Classification task
  - We have a corpus tweets as inputs
  - We have a binary outcome, where tweets with positive sentiments have labels of 1 and tweets with negative sentiments have labels of 0.
- First, we need to process raw tweets and extract meaningful features $X$
- Then we can train our logistic regression model to classify tweets as positive or negative

### Vocabulary & Feature Extraction
![Vocabulary and feature extraction example](figures/vocab-and-fe.png)
- A simple way to represent text numerically is to represent it as a vector of dimension $|V|$
  - $|V|$ corresponds to the size of your vocabulary $V$
  - Vocabulary $V$ is the **set** of all **unique** words that appear in your text examples
- To represent a tweet, we put a 1 in the index of words present in the tweet, and 0 in the indices of the words not present in the tweet
- A drawback of this method is that our vector representations of tweets will become more and more sparse as $V$ gets larger
  - This also results in us having to learn **a lot** of model parameters, $n\theta $ parameters, which will slow down training and prediction time

### Negative and Positive Frequencies
Given a corpus of tweets that have been labeled as positive and negative:
![Sentiment labeled tweets example](figures/labeled-tweets-example.png)

We can create a mapping of each word in our dictionary to its appearance frequency in each sentiment

![Sentiment frequency dictionary](figures/sentiment-freq-dict.png)

### Feature Extraction with Frequencies
- As mentioned above, one-hot encoding our entire vocabulary isn't super practical as our vocabulary grows
- We can use the sentiment-frequency dictionary introduced above to encode tweets in dimensions of 3 ($\mathbb{R}^3$). 
- The first entry in this 3D vector is 1, for the bias term that will be used in the logistic regression (more on that later).
- The second entry is the sum of the **PosFreq** values present in the tweet
![Positive frequency encoding](figures/posfreq-encoding.png)
- The third entry is the sum of the **NegFreq** values present in the tweet
![Negative frequency encoding](figures/negfreq-encoding.png)

- This results in the tweet *"I am sad, I am not learning NLP"*, to be encoded as $[1, 8, 11]$

### Preprocessing
A basic checklist for preprocessing text:
1. Eliminate handles and URLs
2. Tokenize the string into words
3. Remove stop words like "and, is, a, on, etc."
4. Stemming: convert every word to its stem ("dancer", "dancing", "danced" becomes "danc")
5. Convert all words to lower case
   
For example, the following tweet "@YMourri and @AndrewYNg are tuning a GREAT AI model at https://deeplearning.ai!!!" after preprocessing becomes
![Preprocessing tweets](figures/preprocessing.png)

### Putting it all together
- We convert raw text into numerical representations by first preprocessing it and then performing feature extraction:
![Putting it all together](figures/prep-overview.png)

- We can then do this for $m$ tweets, giving us a matrix of vector representations of our tweets $X \in \mathbb{R}^{mx3}$
![Matrix representation of tweets](figures/tweets-matrix.png)

- Python pseudocode for implementing this process:
![Python pseudocode](figures/pipeline-psuedocode.png)

### Logistic Regression