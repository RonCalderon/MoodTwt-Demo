import sys
#import twint
import io
from contextlib import redirect_stdout
import re,string
import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

def scrape_tweet(word,filename,save_bool,limit):
    # Config twint
    c = twint.Config()

    # if sys.argv[1] != "undefined":
        # c.Search = sys.argv[1]
    # if sys.argv[2] != "undefined":
        # c.Username = sys.argv[2]
    # if sys.argv[3] != "undefined":
        # c.Since = sys.argv[3]
    # if sys.argv[4] != "undefined":
        # c.Until = sys.argv[4]

    c.Search = word
    c.Near = "Manila"
    c.Limit = limit
    c.Pandas = True
    c.Lang = "tl"
    
    if save_bool:
        c.Store_csv = True
        c.Output = filename + ".csv"
        
    # Redirect printing of [!]
    f = io.StringIO()
    with redirect_stdout(f):
        twint.run.Search(c)
        
    if save_bool:
        return(filename)
        
    else:    
        Tweets_df = twint.storage.panda.Tweets_df    
        #print(Tweets_df)
        return(Tweets_df)
        
#filter section
def strip_links(text):
    text = str(text)
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    text = str(text)
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)
    
def filter_tweet(filename,save_bool):
    if save_bool:
        csv_input = pd.read_csv(filename + ".csv")
    else:
        csv_input = filename
    newcol = []
    for x in csv_input['tweet']:
        tweet = strip_all_entities(strip_links(x))
        
        # Remove all the special characters
        tweet = re.sub(r'\W', ' ', tweet)

        # remove all single characters
        tweet= re.sub(r'\s+[a-zA-Z]\s+', ' ', tweet)

        # Remove single characters from the start
        tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', tweet) 

        # Substituting multiple spaces with single space
        tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)

        # Removing prefixed 'b'
        tweet = re.sub(r'^b\s+', '', tweet)
        
        # Converting to Lowercase
        tweet = tweet.lower()
        
        if tweet == " " or tweet == "\n":
            tweet = np.nan

        newcol.append(tweet)
    csv_input['filtered_tweet'] = newcol
    if save_bool:
        csv_input.to_csv(filename + ".csv", index=False)
        return(pd.read_csv(filename + ".csv",
        lineterminator='\n'))
    else:
        #print(csv_input)
        return(csv_input)
        
def create_stopwords():
    tl = pd.read_csv('stopwords-tl.txt', sep=" ")
    tl = tl['word']
    tl_stopwords =[]

    for x in tl:
        tl_stopwords.append(x)

    new_stopwords=list(stopwords.words("english")+tl_stopwords)
    return new_stopwords
    
def print_top10(vectorizer, model, class_labels):
    print("===Most Important Features===")
    feature_names = vectorizer.get_feature_names_out()
    for i, class_label in enumerate(class_labels):
        importance = model.coef_[i]
        top10 = np.argsort(model.coef_[i])[-20:]
        print("=",class_label,"=")
        for j in top10:
            #print(feature_names[j])
            # summarize feature importance
            for w,v in enumerate(importance):
                if w == j:
                    print('Feature: %s, Score: %.5f' % (feature_names[j],v)) 
        print()
        
def predict_sentiment(text,df,save_bool,filename):
    texttrain= vectorizer.transform(text)
    
    results = model.predict(texttrain)
    confidence = model.predict_proba(texttrain)
    #print(confidence)
    newcol2 = []
    newcol3 = []
    newcol4 = []
    for i,j in zip(results,confidence):
        newcol2.append(i)
        newcol4.append([j[0],j[1],j[2]])
        if i == -1:
            newcol3.append(j[0])
        elif i == 0:
            newcol3.append(j[1])
        else:
            newcol3.append(j[2])
    df['pred_value'] = newcol2
    df['confidence'] = newcol3
    df['confidencefull'] = newcol4
    df['pred_sentiment'] = df.pred_value.apply(lambda x: "negative" if x == -1 else ("positive" if x==1 else "neutral"))#Adding the sentiments column
    if save_bool:
        df.to_csv(filename + ".csv", index=False)   
        return(df)
    else:
        print("===Sentiments for:", filename,"===")
        print(df[['filtered_tweet','pred_sentiment','confidence']])
        print("Most common sentiment:", df.mode()["pred_sentiment"][0])
        print((df['pred_sentiment'].value_counts()/df['pred_sentiment'].count())*100)
        print_top10(vectorizer, model, ["negative","neutral","positive"])
        return(df)
    
def predict_topic(text,save_bool,exists,limit):
    if not exists:
        df = filter_tweet(scrape_tweet(text,text,save_bool,limit),save_bool)
    else:
        df=pd.read_csv(text + ".csv")
        df=filter_tweet(text,save_bool)
    df.drop_duplicates(subset=["filtered_tweet"], keep='last', inplace=True)
    df.dropna(subset=["filtered_tweet"],inplace=True)
    df.dropna(how='all')
    #df.to_csv(text + ".csv", index=False)   
    return predict_sentiment(df['filtered_tweet'],df,save_bool,text)
    
df1 = pd.read_csv('training.csv', dtype='str')
#filter_tweet('training_nokaggle+hate',True)
df1.drop_duplicates(subset=["filtered_tweet"], keep='last', inplace=True)
df1.dropna(subset=["filtered_tweet"],inplace=True)
df1.dropna(how='all')
#df1 = df1.sample(frac = 1)
df1['value'] = df1.sentiment.apply(lambda x: -1 if x == "negative" else (1 if x=="positive" else 0))#Adding the sentiments column
X1 = df1['filtered_tweet']
y1 = df1['value']

###training section###
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, 
                                        test_size = 0.3, random_state=123)
                                        
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
vectorizer = TfidfVectorizer(lowercase=True,stop_words=create_stopwords(),ngram_range = (1,1),tokenizer = token.tokenize)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Multinomial Naive Bayes
# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB()
# model.fit(X_train_vec, y_train)
# mb_score = model.score(X_test_vec, y_test)
# print("Results for Multinomial Bayes with tfidf:",mb_score)
# #print_top10(vectorizer, model, ["negative","neutral","positive"])

#logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
model = LogisticRegression(max_iter=1000,multi_class="multinomial")
model.fit(X_train_vec, y_train)
lr_score = model.score(X_test_vec, y_test)
prediction = model.predict(X_test_vec)
# print('Accuracy:', accuracy_score(y_test, prediction))
# print('F1 score:', f1_score(y_test, prediction,average="weighted"))
# print('Recall:', recall_score(y_test, prediction,average="weighted"))
# print('Precision:', precision_score(y_test, prediction,average="weighted"))
# print('\nClassification report:\n', classification_report(y_test,prediction))
#print( '\n confussion matrix:\n',confusion_matrix(y_train, prediction))
#print("Results for Logistic Regression with tfidf:",lr_score)
#print_top10(vectorizer, model, ["negative","neutral","positive"])

#pickle.dump(model, open('Moodtwt_model.pkl', 'wb'))

#topic,save as csv, exists as file, limit of tweets to scrape
# predict_topic("Ber Months",True,True,3000)
# predict_topic("HenryPH",True,True,3000)
# predict_topic("Leni Robredo",True,True,3000)

# ###frontend section###
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates

st.set_page_config(
    page_title="MoodTwt",
    page_icon="🦚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("""
# MoodTwt: A Bilingual Tagalog-English Twitter Sentiment Analysis App

This app predicts the sentiment of Philippine Tweets!
(Search for now is only choices due to Twitter API being pay-to-use)

""")

st.write("Current accuracy is : " + "{0:.0%}".format(lr_score))

# st.sidebar.markdown("""
  # <style>
    # .css-6qob1r.e1fqkh3o3 {
      # margin-top: -75px;
    # }
  # </style>
# """, unsafe_allow_html=True)

st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.header('User Input Features')

# Collects user input features into dataframe
topic = st.sidebar.radio("Pick the topic of interest:", ("National ID","Manny","Marcos","Hatsune Miku"))
no_tweets = st.sidebar.slider('Number of Tweets:', 100,500,250)
#try:
if topic is not None:
    df = predict_topic(topic,True,True,no_tweets)
    st.header("Sentiments for : " + topic)
    st.subheader("Most common sentiment : "+ df.mode()["pred_sentiment"][0])
    percentages = (df['pred_sentiment'].value_counts()/df['pred_sentiment'].count())*100
    st.bar_chart(percentages)
    
    df["date"] = pd.to_datetime(df["created_at"])
    #x[0] negative x[1] neutral x[2] positive
    #confidence if positive, -confidence if negative, 1-confidence if neutral positive, -1+confidence if neutral negative
    df['position'] = df.confidencefull.apply(lambda x: -x[0] if x[0]>x[1] and x[0]>x[2] else (x[2] if x[2]>x[1] and x[2]>x[0] else (1-x[1] if x[2]>x[0] else -1+x[1])))
    area = df['confidence']
   
    import plotly.express as px  

    plot = px.scatter(df, x="date", y="position",size=area,size_max=10,hover_data=['tweet'],color="pred_sentiment",
                      labels={
                     "date": "Date of Tweet",
                     "position": "Sentiment Confidence",
                     "pred_sentiment": "Predicted Sentiment",
                     "confidence":"Confidence",
                     "tweet":"Tweet"},
                     color_discrete_sequence = ['yellow', 'gray', 'blue'],
                     category_orders={"pred_sentiment": ["positive", "neutral", "negative"]})
    st.header('Sentiments over time')
    st.plotly_chart(plot, use_container_width=True)
    
    # st.pyplot(fig)  
    
    st.header('Tweets')
    
    st.table(df[['date','tweet','pred_sentiment','confidence']])
#except:
    #st.write("No Tweets gathered.")
