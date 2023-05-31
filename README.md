# MoodTwt

By Ron Cedric P. Calderon and Concepcion L. Khan

MoodTwt: An Application for Sentiment Analysis of Bilingual Tagalog-English Trending Twitter Topics Using Softmax Regression

We present MoodTwt: An Application for Sentiment Analysis of Bilingual Tagalog-English Tweets using Softmax Regression. 
Given an input topic, tweets regarding that topic are gathered and labelled whether they have a positive, neutral, or negative sentiment. 
Softmax Regression is also called Multinomial Logistic Regression and was used as the machine learning algorithm. 
With 19,458 tweets as the training data that were manually tagged,
the machine learning model was able to achieve an accuracy of 82.12\%, precision of 82.11\%, recall of 82.13\%, and F1 of 81.97\%, 
well within the 70\%-90\% range of acceptable industry standards. The application is able to display the most common sentiment for each topic, 
with a graph that shows the date the tweet was posted and the confidence of the model of the predicted sentiment of a specific tweet.

Libraries required:
̶t̶w̶i̶n̶t̶ Does not work as of March 22, 2023 due to Elon Musk Twitter's new anti scraping. Needs to be changed soon
pandas
numpy
nltk
sklearn
streamlit

Use pip install
run using "streamlit run search_twitter.py"

̶O̶n̶l̶i̶n̶e̶ ̶v̶e̶r̶s̶i̶o̶n̶ ̶h̶t̶t̶p̶s̶:̶/̶/̶r̶o̶n̶c̶a̶l̶d̶e̶r̶o̶n̶-̶m̶o̶o̶d̶t̶w̶t̶-̶s̶e̶a̶r̶c̶h̶-̶t̶w̶i̶t̶t̶e̶r̶-̶v̶j̶p̶6̶s̶7̶.̶s̶t̶r̶e̶a̶m̶l̶i̶t̶.̶a̶p̶p̶/̶ Does not work as of March 22, 2023 due to Elon Musk Twitter's new anti scraping.