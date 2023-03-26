from fastapi import FastAPI, Request,WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from deepgram import Deepgram
from dotenv import load_dotenv
from typing import Dict, Callable
import os

load_dotenv()

#instatiate fast api
app = FastAPI() 
      
dg_client = Deepgram(os.getenv('DEEPGRAM_API_KEY'))

#define the template folder for jinja template
templates = Jinja2Templates(directory="templates")

def add_suffix(s):
    vowels = ['a', 'e', 'i', 'o', 'u']
    words = s.split()
    for i in range(len(words)):
        if words[i][-1].lower() in vowels:
            words[i] += '-v'
        else:
            words[i] += '-c'
    return ' '.join(words)

# fast socket keeps the connection open (client and fast api)
async def process_audio(fast_socket: WebSocket):
    #gets transcript and sends to bqck to client
    async def get_transcript(data: Dict) -> None: 
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']

            if transcript:
                await fast_socket.send_text(add_suffix(str(transcript)))

    deepgram_socket = await connect_to_deepgram(get_transcript)

    return deepgram_socket


async def connect_to_deepgram(transcript_received_handler: Callable[[Dict], None]):
    try:
        socket = await dg_client.transcription.live({'punctuate': True, 'interim_results': False})
        socket.registerHandler(socket.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))
        socket.registerHandler(socket.event.TRANSCRIPT_RECEIVED, transcript_received_handler)

        return socket
    except Exception as e:
        raise Exception(f'Could not open socket: {e}')
    

# show index html as the default webpage
@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})




#websocket allows sending of messages 
@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    #we need to create a WebSocket endpoint that listens for client connection

    try:
        #establish connection between fast api and deepgram  
        deepgram_socket = await process_audio(websocket)
        # calls proccess audio function
        while True:
            data = await websocket.receive_bytes()
            #connection between our central FastAPI server and Deepgram
            # print(data)
            # print(transcription_change(data))
            deepgram_socket.send(data)
    except Exception as e:
        raise Exception(f'Could not process audio: {e}')
    finally:
        await websocket.close()