import tweepy #https://github.com/tweepy/tweepy
import csv
from Data_scraping import *
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from skimage.transform import resize
import warnings
from imageio import imread as imr
import re
warnings.filterwarnings("ignore")
#Twitter API credentials
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""


def get_all_tweets(screen_name,consumer_key,consumer_secret,access_key,access_secret):
    '''This function gets all the tweets for the given screen name
    input:
    screen_name--> twitter user name
    consumer_key--> provide your consumer_key generated from dev.twitter.com
    consumer_secret--> provide your consumer_secret generated from dev.twitter.com
    access_key--> provide your access_key generated from dev.twitter.com
    access_secret--> provide your access_secret generated from dev.twitter.com
    '''
    #Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []  

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print ("getting tweets before %s" % (oldest))

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print ("...%s tweets downloaded so far" % (len(alltweets)))

    #transform the tweepy tweets into a 2D array that will populate the csv 
    #outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
    #write the csv  
    #with open('%s_tweets.csv' % screen_name, 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(["id","created_at","text"])
    #    writer.writerows(outtweets)
    with open('%s_tweets.txt' % screen_name, 'w') as f:
        for i in outtweets:
            f.write(i[-1]+'\n')
    return outtweets

def get_tweets_for_all(names):
    '''This function downloads twitter files for all the names listed
    input:
    names--> list of names
    '''
    assert isinstance(names,list) & all(isinstance(i,str) for i in names),"input names not as expected"
    for i in names:
        get_all_tweets(i)
        
def cleaned_up_text(fname,name):
    '''This function prunes the dataset to access important words
    input:
    fname--> file name
    output:
    cleaned up text data as string
    '''
    assert isinstance(fname,str),"file name is not a string"
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    shortword = re.compile(r'\W*\b\w{1,5}\b')
    with open(fname,'r') as content_file:
        content = content_file.read()
        text=re.sub(r'http\S+', '', content)
        text=re.sub(r'%s'%name,'', text)
        text=shortword.sub('',text)
        text=emoji_pattern.sub('',text)
    return text
import numpy as np
def rgb2gray(rgb):
    '''This function creates an rgb 2 gray image
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


import enchant
import pandas as pd
def find_diff(wordcloud,wordcloud3):
    d=enchant.Dict("en-US")
    diff = dict()
    for word in wordcloud.words_.keys():
        if (word in wordcloud3.words_) and d.check(word) :
            diff[word] = wordcloud.words_[word] - wordcloud3.words_[word]
    diff=pd.Series(diff)
    diff=diff.sort_values()
    diff = pd.Series.abs(diff)
    return diff

def file_info(fname):
    with open(fname) as infile:
        lines=0
        words=0
        characters=0
        for line in infile:
            wordslist=line.split()
            lines=lines+1
            words=words+len(wordslist)
            characters += sum(len(word) for word in wordslist)
    return lines,words,characters

def makeImage(text,name,mask):
    '''THis function creates a word cloud for the given text and mask and stores it in name file
    '''
    wc = WordCloud(background_color="white", max_words=1000, mask=mask2)
    # generate word cloud
    wc.generate(text)
    wc.to_file("%s.png"%name)
    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
def create_mask(im_name):
    '''This function creates a mask for creating wordcloud for im_name
    input:
    im_name--> Jpeg file format image (string)
    '''
    mask2=imr(im_name)
    mask2=rgb2gray(mask2)
    mask2[mask2>250]=255
    mask2[mask2<250]=0
    mask2=resize(mask2,(1200,1200))
    return mask2
def makeImage_freq(text,name):
    '''
    '''
    assert isinstance(text,dict)
    mask2=imr('costa_rica.jpg')
    mask2=rgb2gray(mask2)
    mask2[mask2>254]=255
    mask2[mask2<254]=0
    mask2=resize(mask2,(1200,1200))
    wc = WordCloud(background_color="white", max_words=1000, mask=mask2)
    # generate word cloud
    wc.generate_from_frequencies(text)
    wc.to_file("%s.png"%name)
    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
from google.cloud import translate
import os
import time
# Instantiates a client

def translate_file(fname):
    translate_client = translate.Client()
    avg=os.path.getsize(fname)
    n=avg//19000
    print(n)
    with open(fname,'r') as file,open('translated_%s' %fname,'w') as fileT:
        for i in range(n):
            print(i)
            text=file.read(19000)
            text=translate_client.translate(text,target_language='en')
            fileT.write(text['translatedText'])
if __name__ == '__main__':
    data1=cleaned_up_text('translated_Laura_ch_tweets.txt','Laura_Ch')
    data2=cleaned_up_text('translated_CarlosAlvQ_tweets.txt','CarlosAlvQ')
    data3=cleaned_up_text('translated_luisguillermosr_tweets.txt','luisguillermosr')
    wordcloud1 = WordCloud(max_words=35628500,mask=mask2,background_color='white').generate(data1)
    wordcloud3 = WordCloud(max_words=35628500,mask=mask2,background_color='white').generate(data3)
    diff=find_diff(wordcloud1,wordcloud3)
    makeImage_freq(dict(diff[0:60]),"luisguiller")
    makeImage_freq(dict(diff[-60:]),"laura_ch")