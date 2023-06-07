from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd

from utils.utilities import api_key
from utils.utilities import assemble
from utils.utilities import video_details

from keras.models import load_model

app = Flask(__name__)

MODEL_PATH = 'models/final_model.h5'

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':

        model = load_model(MODEL_PATH)

        video_id = request.form['user_news']
        video_id = video_id.split('/')[-1]

        video_title, channel_name = video_details(video_id,api_key)


        output = assemble(video_id, api_key, model)

        out1 = ''
        out2 = ''
        if output[0] > 0:
            out1 = 'Hateful Video'
        else:
            out1 = 'Not hateful video'
        
        if output[1] > 0.7:
            out2 = 'Hateful comments'
        else:
            out2 = 'Supportive comments'
 
            
        #final_output = 'Video Title : '+ video_title + '\nChannel Name: ' + channel_name + '\n' + out1 + '\n' + out2


        return render_template('index.html', video_title = video_title, channel_name = channel_name, out1 = out1, out2 = out2)
    return render_template('index.html')


app.run(debug = True)
