from utils import *
from flask import Blueprint,render_template,request
views = Blueprint("views",__name__)

@views.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('textarea')
        if(not text):
           return render_template('index.html')
        
        naive_bayes_model = load('models/naive_bayes_model_lite.joblib') 
        tfidf_model = load('models/tfidf_vectorizer_model_lite.joblib')
        queries = [text]    
        results = predict(queries,naive_bayes_model,tfidf_model)
        print(results[0]) 
        return render_template('index.html',results=results[0][1],text = text)
    return render_template('index.html')

