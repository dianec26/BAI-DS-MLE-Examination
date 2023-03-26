import torch
from transformers import AutoTokenizer, AutoModel

class TextEmbedder:
    #can change the model just by changing the model name
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        # initialize the model by using the set_model function
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.set_model(model_name)

    def get_model(self, model_name):
        # Loads a general tokenizer and model using 
        # model_name argument
        # returns the model amd tokenizer
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    def set_model(self, model_name):
        #instantiate the model and tokenizer to the class object
        self.model, self.tokenizer = self.get_model(self.model_name)


    def _mean_pooling(self, model_output, attention_mask):
        #  """
        # takes into account the attention layer 
        # outputs a mean pooling layer
        # """
        token_embeddings = model_output[0]
        
        input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
        )
        
        pool_emb = (
        torch.sum(token_embeddings * input_mask_expanded, 1) 
        / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )
        
        return pool_emb
    
    def get_embeddings(self, questions, batch=16):
        # embeds the list of questions 
        # by tokenizing and peforming mean pooling 
        # outputs the embeding vectors of the list of question
        question_embeddings = []
        for i in range(0, len(questions), batch):
        
            # Tokenize sentences
            encoded_input = self.tokenizer(questions[i:i+batch], padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform mean pooling
            batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            question_embeddings.append(batch_embeddings)
        
        question_embeddings = torch.cat(question_embeddings, dim=0)
        return question_embeddings

class QAMaster:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
    #initialize all the parts of the QA
        # answers
        # questions
        # question_emberrdinsg
        self.ans = None
        self.qs = None
        self.q_embeddings = None
        self.embedder = TextEmbedder(model_name=model_name)

    def set_context(self, questions, answers):
        # in init ans, qs and q_embedding were none 
        # so now we are defining it using set context functio
        """
        args:
            ans =  list of answers
            qs  = list of questions
        """
        self.ans = answers
        self.qs = questions
        self.q_embeddings = self.get_q_embeddings(questions)
    
    def get_q_embeddings(self, qs):
        """
        Transforms the embeddings for the questions in 'context'

        Args
        - qs: quesion

        and initialize the embbeded questions
        """
        #embed the questions
        q_embeddings = self.embedder.get_embeddings(qs)
        q_embeddings  = torch.nn.functional.normalize(q_embeddings, p=2, dim=1)
        return q_embeddings.transpose(0,1)

    def cosine_similarity(self, qs, batch=16):
        """
        Gets the cosine similarity between the new questions and the 'context' questions.
        
        Args:
        qs (`list` of `str`):  List of strings defining the questions to be embedded
        batch (`int`): Performs the embedding job 'batch' questions at a time
        """
        q_embeddings = self.embedder.get_embeddings(qs, batch=batch)
        q_embeddings = torch.nn.functional.normalize(q_embeddings, p=2, dim=1)
        
        #q_embeddings the new question embedding
        #self.q_embeddings the
        cosine_sim = torch.mm(q_embeddings, self.q_embeddings)
        
        return cosine_sim

    def extract_answer(self,qs, batch=16):
        """
        pre requisite to use this function, you must define the context first 
        else you are not comparing the input question to anything
        """

        similarity = self.cosine_similarity(qs,batch=batch)

        res = []
        ctr=0
        # why is there a for loop? cause there could be more than 1 question
        for x in similarity:
            best_ix = x.argmax()
            best_q = self.qs[best_ix]
            best_a = self.ans[best_ix]

            res.append(
                {
                'orig_q':qs[ctr],
                'best_q':best_q,
                'best_a':best_a,
                }
            )
            
            ctr+=1
            
        return res