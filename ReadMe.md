
# Lyrics-Inducer

An AI model used to generate the lyrics for a set of words given as the input, using LSTM and NLP on Tensorflow framework.
<div align="center" style="display:block;margin: 0 auto;">
<image src='res.gif' ></image>
</div>


You can go through the Medium blog for a detailed explaination: [here](medium.com/@mohanqwerty5/lyrics-generator-using-lstm-on-tf-2-0-3baf524129b0)	



## Let's begin 

### Required Libraries

- tensorflow: As the framework and to provide the essentials of Keras. 
- numpy:  Array-processing package.
- pandas: For importing data of various file formats.
- matplotlib: Visualization package. 
- LSTM: Neural Network for model sequencing.



###  Dataset Details
The dataset is a self arranged dataset in (.txt) format, framed by gathering and consolidating the lyrics.

Songs included:

-  bad guy -  the box - shape of you - all of me - failing - mine bazzi - heavy - hot girl bummer - the take - myself - skechers - do re mi - moonlight - blinding lights - goosebumps - chal-bombay - rockstar - starboy - cradles - roxanne.


### AI_App (flask files)
```python
-templates
	-index.html:  contains the html + css code.
-app.py:  application root file to call model and run flask.
-dataset.txt:  dataset used.
-model.h5:  keras converted trained model.
-Prockfile:  for heroku deployment.
-requirement.txt: contains required libraries.

  
```
### Steps to run the flask app:
- install flask
```python
pip instal flask
```
- clone the repo:
```python
https://github.com/mohanqwerty5/Lyrics-Inducer2.git
``` 
- go to the app directoy in terminal:
```python
cd AI_App
```
- launch the server
```python
python app.py
```


###  Conclusion 
 Lyrics-Inducer using NLP and LSTM, the model is more like predicting the NEXT WORD according to previous set of words, hence not much accurate for predicting longer sentences, do try it. There are plenty of applications possible using the LSTM and NLP, do try.


