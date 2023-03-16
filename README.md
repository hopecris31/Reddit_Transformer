# ChatRPT(Reddit Pretrained Transformer)

###### Goal: To fine-tune GPT-2 with a Reddit comment corpus.  When the program user enters the beginning of a phrase or sentence, the program should finish the phrase or sentence in a way that "sounds like a Redditor" (or at least an answer that Reddit would have probably given)

## Phase 1: Loading the Dataset
We needed to first load our dataset into our project.  We chose the Hugging Face Reddit Comments Corpus, which is a corpus that contains millions of reddit comments. Each comment is labeled with multiple informative tags, including the User, Subreddit, Content, a Summary, and more.  We chose to just use the subreddit and content to fine-tune our model with with focus on the Subreddit and Content columns.
