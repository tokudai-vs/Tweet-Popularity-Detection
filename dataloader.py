def gettweetfromid(tweet_id):
    CONSUMER_KEY = ""
    CONSUMER_SECRET = ""
    OAUTH_TOKEN = ""
    OAUTH_TOKEN_SECRET = ""

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)

    tweet = api.get_status(id_of_tweet)
    print(tweet.text)
