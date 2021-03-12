
# coding: utf-8

# In[14]:


import pickle
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
import pandas as pd


# In[15]:


file = open('model.pkl', 'rb')
result = pickle.load(file)
file.close()


# In[23]:


@app.route('/')
def hello_world():
    return render_template('show.html', pred=result[0],acc = result[1],rmse = result[2])


# In[25]:


if __name__ == "__main__":
        app.run(debug=True)

