import pandas as pd
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import pickle
from sklearn import preprocessing
from werkzeug.utils import secure_filename
import matplotlib
from uuid import uuid4
import os
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'super secret key'

UPLOAD_FOLDER = '/Users/EleonorK/Git/github.com/projet_python_molecule/flask/file'
RESULT_FOLDER = '/Users/EleonorK/Git/github.com/Eprojet_python_molecule/flask/result'
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/api/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No X_test provided')
        return redirect('/')
    app.logger.info(request.files)
    upload_files = request.files.getlist('file')
    app.logger.info(upload_files)
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    upload_id = str(uuid4())
    if not upload_files:
        flash('No selected file')
        return redirect('/')
    for file in upload_files:
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect('/')
        if file and allowed_file(file.filename):
            # create new folder for each upload
            os.makedirs(os.path.join(
                UPLOAD_FOLDER, upload_id), exist_ok=True)
            file.save(os.path.join(
                f'{UPLOAD_FOLDER}/{upload_id}', secure_filename(file.filename)))
            # Retirect to result page
    return redirect(f'/api/predict/result/{upload_id}')

@app.route('/api/predict/result/<upload_id>')
def result(upload_id):
    # Import X_train and Y_train from the ./models/data folder
    X_train = pd.read_csv("./data/X_train.csv",sep="\t")
    y_test = pd.read_csv("./data/y_test.csv",sep="\t")
    y_train = pd.read_csv("./data/y_train.csv",sep="\t")
    # Import X_test and Y_test
    X_test = np.loadtxt(
        f'{UPLOAD_FOLDER}/{upload_id}/X_test.csv', delimiter=',')
    
    # scaling
    scaler=preprocessing.StandardScaler().fit(X_train)
    X_train_scale=scaler.transform(X_train)
    X_test_scale=scaler.transform(X_test)

    # upload the models
    svm = pickle.load(open('./models/svm.pkl','rb'))
    knn = knn = pickle.load(open('./models/knn.pkl','rb'))
    decision_Tree = pickle.load(open('./models/Decision_Tree.pkl','rb'))
    adaBoost = pickle.load(open('./models/adaBoost.pkl','rb'))
    rdm_forest = pickle.load(open('./models/rdm_forest.pkl','rb'))

    # Get the results
    svm_score = svm.score(X_test_scale, y_test)
    knn_score = knn.score(X_test_scale, y_test)
    decision_Tree_score = decision_Tree.score(X_test_scale, y_test)
    adaBoost_score = adaBoost.score(X_test_scale, y_test)
    rdm_forest_score = rdm_forest.score(X_test_scale, y_test)

    # Get the predictions
    svm_predict = svm.predict(X_test_scale)
    knn_predict = knn.predict(X_test_scale)
    decision_Tree_predict = decision_Tree.predict(X_test_scale)
    adaBoost_predict = adaBoost.predict(X_test_scale)
    rdm_forest_predict = rdm_forest.predict()

    # Create the result folder
    os.makedirs(os.path.join(
        RESULT_FOLDER, upload_id), exist_ok=True)
    # Save each model's predictions
    svm_predict_url = f'{RESULT_FOLDER}/{upload_id}/svm_grid_predict.csv'
    np.savetxt(svm_predict_url,
               svm_predict, delimiter=',')
    knn_predict_url = f'{RESULT_FOLDER}/{upload_id}/knn_grid_predict.csv'
    np.savetxt(knn_predict_url,
               knn_predict, delimiter=',')
    decision_Tree_predict_url = f'{RESULT_FOLDER}/{upload_id}/decision_Tree_grid_predict.csv'
    np.savetxt(decision_Tree_predict_url,
               decision_Tree_predict, delimiter=',')
    adaBoost_predict_url = f'{RESULT_FOLDER}/{upload_id}/adaBoost_grid_predict.csv'
    np.savetxt(adaBoost_predict_url,
               adaBoost_predict, delimiter=',')
    rdm_forest_predict_url = f'{RESULT_FOLDER}/{upload_id}/rdm_forest_grid_predict.csv'
    np.savetxt(rdm_forest_predict_url,
               rdm_forest_predict, delimiter=',')
    gradient_boosting_prediction_url = f'{RESULT_FOLDER}/{upload_id}/gradient_boosting_predictions.csv'

