DATASET DESCRIPTION

The files have been moved to folders. PAths may need to be changed.

dataset_final.csv
Id: Unique identifier for each tweet
Text: Text of each tweet
Retweets: Cascade of number of retweets in each time slot

explicit_features.csv
text_length: Length of text in each tweet
url_count: number of urls in  tweet
followers_count: number of followers of the person 
All the values are normalized using min-max normalization

final_data_cleaned_classification_with_classes.csv
Id: Unique identifier of each tweet
Text: Text of tweet (not used)
1 min: retweets upto 1 min
2 min: retweets upto 2 min
3 min: retweets upto 3 min
5 min: retweets upto 5 min
10 min: retweets upto 10 min
30 min: retweets upto 30 min
1 hour: retweets upto 1 hour
2 hours: retweets upto 2 hour
3 hours: retweets upto 3 hour
5 hours: retweets upto 5 hour
7 hours: retweets upto 7 hour
10 hours: retweets upto 10 hour
12 hours: retweets upto 12 hour
18 hours: retweets upto 18 hour
24 hours: retweets upto 24 hour
36 hours: retweets upto 36 hour
48 hours: retweets upto 48 hour
Labels: Labels generated using no of retweets to find tweets with similiar number of retweets


tokenizer-store.json
tokenizer pre-trained on the corpus

model.py
Python file with the complete code for the model, training and testing

nlp.yml
Conda environment file

TweetPopularityPrediction using DeepLearning
Python notebook of model.py