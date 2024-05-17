from utils import *
from flask import Blueprint,render_template,request
views = Blueprint("views",__name__)

naive_bayes_model = None
tfidf_model = None

@views.route('/',methods=['GET','POST'])
def index():
    global naive_bayes_model
    global tfidf_model
    if request.method == 'POST':
        text = request.form.get('textarea')
        if(not text):
           return render_template('index.html')
        
        if naive_bayes_model is None:
            naive_bayes_model = load('models/naive_bayes_model_lite.joblib') 
        if tfidf_model is None:
            tfidf_model = load('models/tfidf_vectorizer_model_lite.joblib')
        
        queries = [text]    
        results = predict(queries,naive_bayes_model,tfidf_model)
        print(results[0]) 
        return render_template('index.html',results=results[0][1],text = text)
    return render_template('index.html')

