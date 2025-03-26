#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model & Scaler
kmeans = joblib.load('netflix_kmeans.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    release_year = int(request.form['release_year'])
    is_tv_show = int(request.form['is_tv_show'])
    genres = request.form.getlist('genres')  # List of selected genres
    
    # Create DataFrame
    input_data = pd.DataFrame({
        'release_year': [release_year],
        'is_tv_show': [is_tv_show]
    })
    
    # One-hot encode genres (same as training)
    for genre in all_genres:  # Replace with your actual genres
        input_data[genre] = 1 if genre in genres else 0
    
    # Scale & Predict
    X_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(X_scaled)[0]
    
    return f"This content belongs to Cluster {cluster}"

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:






# In[ ]:




