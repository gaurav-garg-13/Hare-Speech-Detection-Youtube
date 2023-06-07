from youtube_transcript_api import YouTubeTranscriptApi
import re
import requests
import os
import googleapiclient.discovery
import googleapiclient.errors

import tensorflow.keras as keras
import pandas as pd
import nltk
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot


api_key = 'AIzaSyC9NunPEERpDpeKqywJ4NNA0p2OmLumLZY'

ps = PorterStemmer()
voc_size = 100000
embedding_vector_features=128
sent_length=200

def video_details(video_id, api_key):
    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)
    video_details = youtube.videos().list(part='snippet', id=video_id).execute()
    video_title = video_details['items'][0]['snippet']['title']
    channel_name = video_details['items'][0]['snippet']['channelTitle']

    return video_title, channel_name


def transcript(video_id):
    outls = []
    tx = YouTubeTranscriptApi.get_transcript(video_id,languages = ['en'])
    for i in tx:
        outtxt = (i['text'])
        outls.append(outtxt + ' ')
        
    final = ''.join(outls)
    text =  re.sub(r'[\W_]+', ' ', final)
    #text = re.sub('[^a-zA-Z]', ' ', test
    
    text = text.lower()
    text = text.split()
    
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    
    return final



def comments(video_id, api_key, max_results = 200):
    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)
    
    try:
        # Request the top comments for the video
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            order="relevance",
            maxResults=max_results
        ).execute()

        # Extract the comments
        comments = []
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            
            comment =  re.sub(r'[\W_]+', ' ', comment)

            comment = comment.lower()
            comment = comment.split()

            comment = [ps.stem(word) for word in comment if not word in stopwords.words('english')]
            comment = ' '.join(comment)
            
            comments.append(comment)

        return comments

    except googleapiclient.errors.HttpError as e:
        print(f"An error occurred: {e}")
        return []
    

def break_text(text, chunk_size=200):
    words = text.split() 
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) == chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def prediction(text,inp_model):
    
    
    test_one_hot = one_hot(text,100000)
    
    l = len(test_one_hot)
    if l<200:
        pad = 200 - l
        a = [0 for x in range(pad)]
        a.extend(test_one_hot)
        test = np.array(a)
        test = test.reshape(-1,200)
    else:
        test_one_hot = test_one_hot[:200]
        test = np.array(test_one_hot)
        test = test.reshape(-1,200)
        
    ans = inp_model.predict(test)

    return ans


def transcript_prediction(s_corpus, model):
    sentiment_list = []
    for i in s_corpus:
        sentiment = prediction(i, model)
        sentiment_list.append(sentiment)
    return sentiment_list


def comment_prediction(comments,model):
    sentiment_list = []
    for i in comments:
        sentiment = prediction(i, model)
        sentiment_list.append(sentiment)
    return sentiment_list


def assemble_script(video_id, model):
    script = transcript(video_id)
    script_corpus = break_text(script)
    sent_scores =  transcript_prediction(script_corpus, model)
    
    hate = []
    positive = []
    for i in sent_scores:
        if i>0.7:
            hate.append((0.7 - i) / 0.3)
        else:
            positive.append((0.7 - i) / 0.7)
    
    # If no hate sections are present
    if len(hate) == 0:
        avg = sum(positive)/len(positive)
        return avg
    
    # If no positive sections are present
    elif len(positive) == 0:
        avg = sum(hate)/len(hate)
        return avg
    
    # If there are more than twice as many positive than hate
    if len(positive) >= 2*len(hate):
        avg = sum(positive)/len(positive)
        return avg
    
    hate_avg = sum(hate)/len(hate)
    posi_avg = sum(positive)/len(positive)
    
    if -hate_avg >= posi_avg:
        return hate_avg
    
    else:
        return posi_avg
    


def assemble_comment(video_id, api_key, model):
    comment = comments(video_id, api_key)
    
    sent_list = comment_prediction(comment, model)
    
    hate = []
    positive = []
    for i in sent_list:
        if i>0.7:
            hate.append(i)
        else:
            positive.append(0.7)
            
    # If no hate comments are present
    if len(hate) == 0:
        avg = sum(positive)/len(positive)
        return avg
    
    # If no positive comments are present
    elif len(positive) == 0:
        avg = sum(hate)/len(hate)
        return avg
    
    # If there are more than 1.5 as many positive comments than hate
    if len(positive) >= 1.5*len(hate):
        avg = sum(positive)/len(positive)
        return avg
    
    # If there are more than 1.5 as many hate comments
    elif len(hate) >= 1.5*len(positive):
        avg = sum(hate)/len(hate)
        return avg
    
    return sum(sent_list)/len(sent_list)



def assemble(video_id, api_key, model):
    t_ans = assemble_script(video_id, model)
    c_ans = assemble_comment(video_id, api_key, model)
    
    return t_ans, c_ans