from flask import Flask ,render_template
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing import sequence
import emoji

model = pickle.load(open('model.pkl', 'rb'))
wordToIndex=pickle.load(open('wordToindex.pkl','rb')) 
emoji_dict={'0': ":heart:",'1': ":baseball:",'2': ":smile:",'3': ":disappointed:",'4': ":fork_and_knife:"}
#model = pickle.load(open('model.pkl', 'rb'))
datas=np.array(['I love you'])

def padding(data):
    data=data[0].split()
    print(data)
    temp_list=[]
    for j in data:
        if wordToIndex.get(j):
            temp_list.append(wordToIndex[j])
        else:
            temp_list.append(0)
    padded_seq=sequence.pad_sequences([temp_list],15,padding='post')
    padded_list=np.array(padded_seq)
    return padded_list



padded_data=padding(datas)
	# return render_template('result.html',prediction = my_prediction)
prediction = model.predict(padded_data)
a=np.argmax(prediction)
s=emoji.emojize(emoji_dict[str(a)],use_aliases=True)
print(datas.shape)
print("datas",datas)
print(s)









