# BAI DS/MLE Examination
 DS/MLE examination application


## Text classifcation
- location: 1_text_classfication
- description: Finetuned an existing model huggingface called distilbert-base-uncased (with the task of sentiment analysis), to a hate speech identification dataset
- just need to run the notebook in colab and so you can use th GPU for finetunning the model
- result: 91% accuracy
- if using colab: click runtime > change runtime type > GPU
- caveat: ran out of memory and used the memory when hyperparameter tunning but got the best model from that

## Audio classification
- location: 2_audio_classfication 
- description: using pretrained model to ```facebook/wav2vec2-base``` to finetune on speech commands data with the possible result of:
(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow', 'backward', 'forward', 'follow', 'learn', 'visual', '_silence_'])
- result:
- if using colab: click runtime > change runtime type > GPU
- **important note:**the model ran for 3 hours maybe next time less model


## QA
- Location: 3_QA
- Description: The model was based on squad (Standord Questiion and Answering dataset), and deploy it via docker and FastAPI. 
- Task methodology: 
    1. Setting the context (question and asnwers) by picking a topic in squad
    2. When a user asks a question, we compare the user_genererated questions to the existing questions is squad. 
    3. Get the answer from the most similar question (Squad vs User generated question)
- files:
    - essentials.py: contains 2 classes
        - TextEmbedder: used for getting and setting embbeding of the text
        - QAMaster: Using the text embedding
    - main.py
- caveat: did not test this because there was an issue with docker


## Transcription:
- 
- **important note**: 
    1. create a 4_Transcription/.env file containing the api key (specifically in this format) DEEPGRAM_API_KEY = "insert key here"
    2. pip
