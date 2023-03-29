# BAI DS/MLE Examination
 DS/MLE examination application


## Text classifcation
- location: 1_text_classfication
- description: Finetuned an existing model huggingface called distilbert-base-uncased (with the task of sentiment analysis), to a hate speech identification dataset
- just need to run the notebook in colab and so you can use th GPU for finetunning the model
- if using colab: click runtime > change runtime type > GPU

## Audio classification
- location: 2_audio_classfication 
- description: using pretrained model to ```facebook/wav2vec2-base``` to finetune on speech commands data with the possible result of:
(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow', 'backward', 'forward', 'follow', 'learn', 'visual', '_silence_'])
- if using colab: click runtime > change runtime type > GPU

## QA
- location: 3_QA
- 
- caveat: did not test this


## Transcription:
- 