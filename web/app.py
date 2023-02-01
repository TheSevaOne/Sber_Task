from flask import Flask,render_template,jsonify ,request
import sqlite3
from datetime import date, datetime
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import utils
import shutil
from PIL import Image as im
from  torchvision.utils import save_image
app = Flask(__name__)   
mean_in=(0.485, 0.456, 0.406),
std_in=(0.229, 0.224, 0.225)
SIZE=(225,450)
conn = sqlite3.connect('static/data.db', check_same_thread=False)
curs = conn.cursor()
conn.row_factory = sqlite3.Row
data=curs.execute("SELECT * FROM history")
for column in data.description:
    print(column[0])
model=utils.model_init()
print('loaded')
@app.route('/',methods=["GET"])
def main():
    time=datetime.now()
    utils.cleaner()
    file=utils.chooseRandomImage('../train_images')
    shutil.copy('../train_images/'+file, 'static/uploads')
    input_tensor = transform('static/uploads/'+file,'web')
    prediction = get_prediction(input_tensor)
    answer,rle=utils.runner(prediction,"web")
    f=utils.process_out(file,rle)
    
    curs.execute("INSERT INTO history ('time','defect') VALUES (?, ?)",(str(time),str(answer)))
    conn.commit()
    curs.execute("SELECT * FROM history ORDER BY id")

    rows = curs.fetchall()
    
    return render_template('index.html',det=f,image="static/uploads/"+file,db_data=rows,)


@app.route('/api', methods=['POST'])
def predict():
    answer=[]
    time=datetime.now()
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform(file,'api')
            prediction = get_prediction(input_tensor)
            answer,_=utils.runner(prediction,"api")
            curs.execute("INSERT INTO history ('time','defect') VALUES (?, ?)",(str(time),str(answer)))
        return jsonify({str(time):answer}) 


def get_prediction(input_tensor):
    print(input_tensor.to("cuda:0").size()) 
    outputs = model.forward(input_tensor.to("cuda:0")) 
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu().numpy()
    return outputs


def transform(file,type):
    if type=='api':
        img = np.asarray(bytearray(file.read()))
        img = cv2.imdecode(img, -1)     
    if type =='web':
        img= cv2.imread(file) 
    img=cv2.resize(img,(SIZE[1],SIZE[0])) 
    transform = transforms.ToTensor()  
    return transform(img).unsqueeze_(0)
  

app.run()