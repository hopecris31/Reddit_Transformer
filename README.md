# ChatRPT(Reddit Pretrained Transformer)

###### Goal: To fine-tune GPT-2 with a Reddit comment corpus.  When the program user enters the beginning of a phrase or sentence, the program should finish the phrase or sentence in a way that "sounds like a Redditor" (or at least an answer that Reddit would have probably given). This Project was inspired by our N-Gram Language Model project, along with the Fine-Tuning a Pre-Trained Model class programming activity.

## 🤩Phase 1: Loading the Dataset
We needed to first load our dataset into our project.  We chose the Hugging Face Reddit Comments Corpus, which is a corpus that contains millions of reddit comments. Each comment is labeled with multiple informative tags, including the User, Subreddit, Content, a Summary, and more.  We chose to just use the subreddit and content to fine-tune our model with with focus on the Subreddit and Content columns.

## 🥳Phase 2: Tokenization
With the dataset loaded into our program, we then needed to tokenize it.  Tokenizing breaks down a body into smaller subunits, or tokens, that represent words, phrases, or any meaningful grouping of words seen in the text.  This is so the model can analyze the text better during the Mapping step.  We used the Hugging Face AutoTokenizer, which automatically selected the best tokenizer for us based on our model and program criteria.  

## 🥰Phase 3: Mapping
The next step was to map this tokenized data. We called the Hugging Face **map()** function to do this. This process would turn the data tokens into a numerical or vector representations, so that our model would be able to process the data and analyze information about words in the text.

## 🫣Phase 4: Initializing the Trainer
Once again, we use the Hugging Face library to train our model. But before we can do that, we must set up some parameters for how our model will be trained.  In the **training_args** function, we specify our evaluation strategy (we used epochs), our learning rate, and weight decay.  These were then passed into the **trainer** object, which is where we initialized our model (GPT-2), passed in our training arguments, and defined our training and evaluation sets.

*note: Because of limited computing resources (GPU, RAM, Storage), we were only able to use 5% of the entire reddit dataset.  While this is not realistic or ideal, it was sufficient to demonstrate the ability of our project.  We therefore only used a train dataset*

## 🥹Phase 5: Training
Once everything was passed in, we could train the model with the **trainer.train()** line.
