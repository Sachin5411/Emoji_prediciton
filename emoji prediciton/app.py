from flask import Flask ,render_template
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing import sequence
import emoji

app = Flask(__name__)

wordToIndex=pickle.load(open('wordToindex.pkl','rb')) 
emoji_dict={'0': ":heart:",'1': ":baseball:",'2': ":smile:",'3': ":disappointed:",'4': ":fork_and_knife:"}

def padding(data):
    
    data=data[0].split()
    temp_list=[]
    for j in data:
        if wordToIndex.get(j):
            temp_list.append(wordToIndex[j])
        else:
            temp_list.append(0)
    padded_seq=sequence.pad_sequences([temp_list],15,padding='post')
    padded_list=np.array(padded_seq)
    print("padded list shape",padded_list.shape)
    return padded_list

# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    comment = request.form['comment']
    data = [comment]
    print("comment is ",comment)
    data=np.array(data)
    print("data is ",data)
    print("data type is ",type(data))
    print('data shape is ',data.shape)
    padded_data=padding(data)
    print("padded data is ",padded_data)
    print("padded data shape is ",padded_data.shape)
    model = pickle.load(open('model.pkl','rb'))
    prediction = model.predict(padded_data)
    print("prediction is ",prediction)
    a=np.argmax(prediction)
    print("a is ",a)
    s=emoji.emojize(emoji_dict[str(a)],use_aliases=True)
    return render_template('result.html', prediction_text=s)

if __name__== '__main__':
    app.run(debug=True)