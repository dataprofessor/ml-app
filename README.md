# mlapp

# Watch the tutorial video

[How to Build a Machine Learning App | Streamlit #13](https://youtu.be/eT3JMZagMnE)

<a href="https://youtu.be/eT3JMZagMnE"><img src="http://img.youtube.com/vi/eT3JMZagMnE/0.jpg" alt="How to Build a Machine Learning App | Streamlit #13" title="How to Build a Machine Learning App | Streamlit #13" width="400" /></a>

# Demo

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/dataprofessor/ml-app/main/ml-app.py)

# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *ml*
```
conda create -n ml python=3.7.9
```
Secondly, we will login to the *ml* environement
```
conda activate ml
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://raw.githubusercontent.com/dataprofessor/ml-auto-app/main/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```
###  Download and unzip contents from GitHub repo

Download and unzip contents from https://github.com/dataprofessor/ml-app/archive/main.zip

###  Launch the app

```
streamlit run app.py
```
