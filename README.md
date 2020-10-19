# Text sentiment changer
 Detects if text has a negative sentimental meaning, change it to positive using Bayes method.

Some basic UI was give to the project. All UI is translated to English (translated from Lithuanian).

<img style="float: LEFT;" src="https://github.com/deibraz-free/Text-sentiment-changer/blob/master/img/2.png">
<img style="float: LEFT;" src="https://github.com/deibraz-free/Text-sentiment-changer/blob/master/img/1.png">
<img style="float: LEFT;" src="https://github.com/deibraz-free/Text-sentiment-changer/blob/master/img/3.png">

## What is it?
First ever Machine Learning system that I've made (over a year ago), for an university project. Uploading older projects that might be useful to some people. It's rather simple, but does its job and is a nice entry point into machine-learning world.

Project uses Bayes algorithm to check if text input text has a positive or negative sentimental value

## How does it work?
- Check if model exists, if not, create new file (check if .pickle file exists);

#### Training processes:
- Use NLTK to download train, test data, stop words, etc;
- Preprocess - remove un-needed characters, lemmatize, tokenize text data;
- Gather data into 4 datasets;
- Shuffle dataset contents around;
- Use Naive Bayes to train a classifier model using prepared data;
- Calculate training accuracy;
- Save model.

#### Prediction processes:
- Preprocess (same as train data) given text;
- Use classifier to classify if positive/negative, if negative continue, if positive return text;
- Scan through text, check which words are negative, use antonyms to change to positive using levenstein distance;

## Project uses:
- Python 3.6;
- NLTK - library to gather the training, testing data, processing some of the data;
- TKinter - for GUI handling;
- Pickle - save/load model.
