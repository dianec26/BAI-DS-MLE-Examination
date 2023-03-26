import uvicorn
from fastapi import FastAPI, Request
from essentials import QAMaster

app = FastAPI()



@app.post("/set_context")
async def set_context(data:Request):
  """
  Fastapi POST method that sets the QA context for search.
  
  Args:
    data(`dict`): Two fields required 'questions' (`list` of `str`)
      and 'answers' (`list` of `str`)
  """
  data = await data.json()
  
  qa_master.set_context(
    data['questions'], 
    data['answers']
  )

  return {"message": "Search context set"}


@app.post("/get_answer")
async def get_answer(data:Request):
  """
  Fastapi POST method that gets the best question and answer 
  in the set context.
  
  Args:
    data(`dict`): One field required 'questions' (`list` of `str`)
  
  Returns:
    A `dict` containing the original question ('orig_q'), the most similar
    question in the context ('best_q') and the associated answer ('best_a').
  """
  data = await data.json()
  
  response = qa_master.extract_answer(data['questions'], batch=1)
  return response


# initialises the QA model and starts the uvicorn app
if __name__ == "__main__":
  qa_master = QAMaster()
  uvicorn.run(app, host="0.0.0.0", port=8000)
