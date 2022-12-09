#!/usr/bin/env python
# coding: utf-8

# In[24]:


from flask import Flask
import pandas as pd
from sklearn import linear_model
from joblib import dump, load
import numpy as np
 
app = Flask(__name__)
 
@app.route('/predict/<a>/<b>/<c>/<d>/<e>/<f>/<g>/<h>/<i>/<j>/<k>/<l>/<m>/<n>/<o>/<p>/<q>/<r>/<s>/<t>/<u>/<v>/<w>/<x>/<y>/<z>/<aa>/<ab>/<ac>/<ad>/<ae>/<af>/<ag>/<ah>/<ai>/<aj>/<ak>/<al>/<am>/<an>/<ao>')
def predict(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao):
    cl = load('cl_svm.modele')
    return str(cl.predict(np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao]).reshape(1,-1)))
 
if __name__ == '__main__':
      app.run(debug=True, host='0.0.0.0', port=8080)


# In[25]:


predict(0.10008072, -0.01579791, -0.48949694, -0.16463526, -0.45810201,
       -0.26493012, -0.74677214, -0.92310165,  1.27394585,  1.29089002,
       -0.49566249,  0.26153396, -0.25397319, -0.66525225,  0.16261691,
        0.99557665,  0.04823455,  0.61935794, -0.08461622, -0.22582502,
       -0.13127007, -1.28190998, -0.16458205, -0.1886341 , -0.42669993,
       -0.15532426,  0.08712823,  0.028477  , -0.15579424, -0.73150319,
       -0.77582124, -0.18345485, -0.58745942, -0.57768413, -0.78023694,
       -0.47805035, -0.4131063 , -0.64691142, -0.51852736, -0.15195182,
       -0.31240471)

