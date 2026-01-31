from flask import Flask, render_template, request, session 
import os
# from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
# import MySQLdb.cursors
import re
import mediapipe as mp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as Kn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
# from yellowbrick.cluster import KElbowVisualizer         
from sklearn.cluster import AgglomerativeClustering, KMeans
import seaborn as sns
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import io
import base64
from mlxtend.plotting import plot_confusion_matrix
import cv2
import  cvzone
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)

UPLOAD_FOLDER='static/upload/'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route('/')
def home(): 
  return render_template('home.html')

@app.route('/numpy')
def numpy(): 
  return render_template('numpy.html')

@app.route('/add_numbers', methods=['POST'])
def add_numbers():
	num1 = int(request.form['num1'])
	num2 = int(request.form['num2'])
	result = num1 + num2
	return render_template('numpy.html', result=result)

@app.route('/create-array', methods=['POST'])
def create_array():
    set1 = request.form['set1']
    set2 = request.form['set2']

    set1_array = [int(num) for num in set1.split(',')]
    set2_array = [int(num) for num in set2.split(',')]
    

    x = np.array([set1_array, set2_array])

    return render_template('numpy.html', x=x)

@app.route('/create-index', methods=['POST'])
def create_index():
    if request.method == 'POST':
        set3 = request.form['set3']
        num_list = [int(n) for n in set3.split(',')]
        y= np.array([num_list])
        num3 = request.form['num3']
        if num3:
            try:
                value = num_list[int(num3)]
                return render_template('numpy.html', value=value, num3=num3, y=y)
            except:
                return render_template('numpy.html', error="Invalid index")
    return render_template('numpy.html')

@app.route('/sort', methods=['POST'])
def create_sort():
    import numpy as np
    set4 = request.form['set4']
    set4_array = [int(num) for num in set4.split(',')]
    sortarray=np.array(set4_array)
    sortvalue=np.sort(sortarray)
    return render_template('numpy.html',sortvalue=sortvalue)

@app.route('/pandas')
def pandas(): 
  return render_template('pandas.html')

@app.route('/createseries', methods=['POST'])
def create_dataseries():
    my_dict = {}
    df1 = request.form['df1']
    df1_list = [ n for n in df1.split(',')]
    data = request.form['data']
    data_list = [ n for n in data.split(',')]
    for i in range( 0,len(df1_list),1):
                   my_dict[df1_list[i]]= data_list[i]              
    

    datas=pd.Series(my_dict,index=df1_list)
    return render_template('pandasoutput.html',datasi=datas.index,datas=datas)
        
@app.route('/csv', methods=['POST'])
def csv():
    # df2 = str(request.form['name1'])
    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    read=pd.read_csv(path)         
    return render_template('pandasoutput.html', tables1=[read.to_html()], titles=[''])
 
 
@app.route('/csvdesc', methods=['POST'])
def csvdesc():
    
    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    read=pd.read_csv(path)
    readdes=read.describe()          
    return render_template('pandasoutput.html', tables2=[readdes.to_html()], titles=[''])

@app.route('/csvinfo', methods=['POST'])
def csvinfo():
   
    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    read=pd.read_csv(path)
    readdd=pd.DataFrame(read)
    readss=readdd.isnull()          
    return render_template('pandasoutput.html', tables3=[readss.to_html()], titles=[''])



@app.route('/seaborn')
def seaborn(): 
  return render_template('seaborn.html')


@app.route('/normaldistribution',methods=['POST'])    
def normal_distribution():    

    n=int(request.form['n'])
    y=int(request.form['y'])

    buffer = io.BytesIO()
    f, ax = plt.subplots(figsize=(5,5))
    if y==1:
        sns.distplot(np.random.normal(size=n))
        ax.figure.savefig(buffer, format="png")
        buffer.seek(0)
        image_memory = base64.b64encode(buffer.getvalue())
        normal = image_memory.decode('utf-8')
    else:
        sns.distplot(np.random.normal(size=n),hist=False)
        ax.figure.savefig(buffer, format="png")
        buffer.seek(0)
        image_memory = base64.b64encode(buffer.getvalue())
        normal = image_memory.decode('utf-8')    

    return render_template("seabornoutput.html",img4=normal) 


@app.route('/binomialdistribution',methods=['POST'])    
def binomial_distribution():
        
    p=int(request.form['p'])
    q=int(request.form['q'])
    r=float(request.form['r'])
    buffer = io.BytesIO()
    f, ax = plt.subplots(figsize=(5,5))
    sns.distplot(np.random.binomial(n=q, p=r, size=p), hist=True, kde=False)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img = image_memory.decode('utf-8')  

    return render_template("seabornoutput.html",img5=img)


@app.route('/poissondistribution',methods=['POST'])    
def poisson_distribution():
    
    a=int(request.form['a'])
    b=int(request.form['b'])
    buffer = io.BytesIO()
    f, ax = plt.subplots(figsize=(5,5))
    sns.distplot(np.random.poisson(lam=b, size=a), kde=False)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img = image_memory.decode('utf-8')

    return render_template("seabornoutput.html",img6=img) 

@app.route('/mplot')
def mplot(): 
  return render_template('mplot.html')


@app.route("/matplotlib_result", methods = ['POST'])
def matplotlib_result():
    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    dataset.save(os.path.join(UPLOAD_FOLDER, filename))
    path = os.path.join(UPLOAD_FOLDER, filename)

    df = pd.read_csv(path)
    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)

    a = int(request.form['first'])
    b = int(request.form['second'])
    x = object_cols[a]
    y = object_cols[b]

    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(df[x], df[y], color='r', marker='.', linestyle='-', linewidth=2, markersize=2)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.grid(color='k', linestyle='--', linewidth='0.5')
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_line = image_memory.decode('utf-8')

    # # Scatter Plot
    f, ax = plt.subplots(figsize=(5,5))
    plt.scatter(df[x], df[y], edgecolors='black')
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_scatter = image_memory.decode('utf-8')

    # Bar Graph
    f, ax = plt.subplots(figsize=(5,5))
    plt.bar(df[x], df[y], color='b', width=0.5)
    plt.ylabel(y)
    plt.xlabel(x)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_bar = image_memory.decode('utf-8')

    # Area Plot
    days = df[x]
    age = df[y]
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot([], [], color='c', label=x, linewidth=5)
    plt.plot([], [], color='g', label=y, linewidth=5)
    plt.stackplot(days, age, colors=['c', 'g'])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_area = image_memory.decode('utf-8')

    return render_template('mplotoutput.html', img_line=img_line, img_scatter=img_scatter, img_bar=img_bar, img_area=img_area)
    

@app.route('/knn')
def knn(): 
    return render_template('knn.html')

@app.route('/output')
def output(): 
    return render_template('output.html')

@app.route("/knn_result", methods = ['POST'])
def knn_result():
    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    dataset.save(os.path.join(UPLOAD_FOLDER, filename))
    path = os.path.join(UPLOAD_FOLDER, filename)
    k = int(request.form['k'])
    test = int(request.form['test'])/100
    target = int(request.form['target'])

    df = pd.read_csv(path)
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    l = LabelEncoder()

    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))

    for i in object_cols_float:
        df[i] = l.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]

    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')

    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    
    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, train_size=(1-test), random_state=0)

    KNN = Kn(n_neighbors=k, weights='distance', metric='minkowski', p= 2 )
    KNN.fit(X_train,Y_train)
    Y_predict = KNN.predict(X_test)
    acc = KNN.score(X_test, Y_test)*100
    fscore = f1_score(Y_test, Y_predict, average='weighted')*100

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = KNN.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_knn = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_KNN = %0.2f' % auc_knn)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')

    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc, corr = corr)


@app.route('/lr')
def lr(): 
  return render_template('lr.html')

@app.route("/linreg_result", methods = ['POST'])
def linreg_result():
    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    dataset.save(os.path.join(UPLOAD_FOLDER, filename))
    path = os.path.join(UPLOAD_FOLDER, filename)

    test = int(request.form['test'])/100
    first = int(request.form['first'])
    second = int(request.form['second'])
    u_value = int(request.form['pr'])

    df = pd.read_csv(path)
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    l = LabelEncoder()

    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))

    for i in object_cols_float:
      df[i] = l.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    X = object_cols[first]
    y = object_cols[second]

    X_train, X_test, y_train, y_test = train_test_split(df[X].values.reshape(-1, 1), df[y].values.reshape(-1, 1),test_size=test, train_size=(1-test), random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Visualizing the Training set results
    f, ax = plt.subplots(figsize=(5,5))
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.xlabel(X)
    plt.ylabel(y)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    trs = image_memory.decode('utf-8')

    # Visualizing the Test set results
    f, ax = plt.subplots(figsize=(5,5))
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.xlabel(X)
    plt.ylabel(y)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    tes = image_memory.decode('utf-8')
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    perc = np.mean(y_test)

    s1 = "Columns preferably not suitable after Linear Regression"
    s2 = "Columns suitable after Linear Regression"
    if rmse > (0.1 * perc):
        stat = s1
    else:
        stat = s2

    pred = regressor.predict([[u_value]])

    return render_template('lroutput.html', trs=trs, tes=tes, mae=mae, mse=mse, rmse=rmse, pred=pred, stat=stat)


@app.route('/dt')
def dt(): 
  return render_template('dt.html')

@app.route("/dt_result", methods = ['POST'])
def dt_result():
    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    dataset.save(os.path.join(UPLOAD_FOLDER, filename))
    path = os.path.join(UPLOAD_FOLDER, filename)
    depth = int(request.form['depth'])
    test = int(request.form['test'])/100
    target = int(request.form['target'])

    df = pd.read_csv(path)
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    l = LabelEncoder()

    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))

    for i in object_cols_float:
        df[i] = l.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]

    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')

    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, train_size=(1-test), random_state=0)

    model = tree.DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')
    
    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = model.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_dt = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_Decision_Tree = %0.2f' % auc_dt)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')

    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc, corr = corr)


@app.route('/rf')
def rf(): 
  return render_template('rf.html')


@app.route("/rf_result", methods = ['POST'])
def rf_result():
    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    dataset.save(os.path.join(UPLOAD_FOLDER, filename))
    path = os.path.join(UPLOAD_FOLDER, filename)
    test = int(request.form['test'])/100
    target = int(request.form['target'])

    df = pd.read_csv(path)
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    l = LabelEncoder()

    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))

    for i in object_cols_float:
        df[i] = l.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]

    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')

    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, train_size=(1-test), random_state=0)

    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    Y_predict = rf.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = rf.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_rf = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_Random_Forest = %0.2f' % auc_rf)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')



    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc, corr=corr)

@app.route('/svm')
def svm(): 
  return render_template('svm.html')
 
@app.route("/svm_result", methods = ['POST'])
def svm_result():
    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    dataset.save(os.path.join(UPLOAD_FOLDER, filename))
    path = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(path)
    test = int(request.form['test'])/100
    target = int(request.form['target'])

    df = pd.read_csv(path)
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    l = LabelEncoder()

    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))

    for i in object_cols_float:
        df[i] = l.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]

    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')

    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, train_size=(1-test), random_state=0)

    svc = SVC(probability=True)
    svc.fit(X_train,Y_train)
    Y_predict = svc.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = svc.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_svm = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_SVM = %0.2f' % auc_svm)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')



    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc, corr=corr)


@app.route('/km')
def km(): 
  return render_template('km.html')

@app.route("/km_result", methods = ['POST'])
def km_result():
    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    dataset.save(os.path.join(UPLOAD_FOLDER, filename))
    path = os.path.join(UPLOAD_FOLDER, filename)
    test = int(request.form['test'])/100
    target = int(request.form['target'])

    df = pd.read_csv(path)
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    l = LabelEncoder()

    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))

    for i in object_cols_float:
        df[i] = l.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]

    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')

    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, train_size=(1-test), random_state=0)

    logr = LogisticRegression()
    logr.fit(X_train, Y_train)
    Y_predict = logr.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = logr.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_lr = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_Logistic_regression = %0.2f' % auc_lr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')

    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc, corr = corr)


@app.route('/aglo')
def aglo(): 
  return render_template('aglo.html')


# @app.route("/agglo_o", methods = ['POST'])
# def agglo_o():
#     dataset = request.files['dataset']
#     filename = secure_filename(dataset.filename)
#     dataset.save(os.path.join(UPLOAD_FOLDER, filename))
#     path = os.path.join(UPLOAD_FOLDER, filename)
#     df = pd.read_csv(path, sep='\t')
#     k = int(request.form['k'])
#     clusters = int(request.form['clusters'])

#     data = df.dropna().copy()

#     data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
#     dates = []
#     for i in data["Dt_Customer"]:
#         i = i.date()
#         dates.append(i)

#     days = []
#     d1 = max(dates)
#     for i in dates:
#         delta = d1 - i
#         days.append(delta)
#     data["Customer_For"] = days
#     data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

#     data["Age"] = 2021 - data["Year_Birth"]
#     data["Spent"] = data["MntWines"] + data['MntFruits'] + data["MntMeatProducts"] + data["MntFishProducts"] + \
#                     data["MntSweetProducts"] + data["MntGoldProds"]
#     data["Living_With"] = data["Marital_Status"].replace(
#         {"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone",
#         "Divorced": "Alone", "Single": "Alone", })
#     data["Children"] = data["Kidhome"] + data["Teenhome"]
#     data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
#     data["Is_Parent"] = np.where(data.Children > 0, 1, 0)
#     data["Education"] = data["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate",
#                                                 "Graduation": "Graduate", "Master": "Postgraduate",
#                                                 "PhD": "Postgraduate"})
#     data = data.rename(
#         columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
#                 "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})
#     to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
#     data = data.drop(to_drop, axis=1)

#     # removing outliers
#     data = data[(data["Age"] < 90)]
#     data = data[(data["Income"] < 600000)]

#     s = (data.dtypes == 'object')
#     object_cols = list(s[s].index)

#     LE = LabelEncoder()
#     for i in object_cols:
#         data[i] = LE.fit_transform(data[i])

#     # Scaling The Copy of Dataset
#     ds = data.copy()
#     cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
#     ds = ds.drop(cols_del, axis=1)
#     scaler = StandardScaler()
#     scaler.fit(ds)
#     scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)

#     # Dimentionality Reduction
#     pca = PCA(n_components=3)
#     pca.fit(scaled_ds)
#     PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1", "col2", "col3"]))

#     buffer = io.BytesIO()
#     # Agglomerative Clustering
#     AC = AgglomerativeClustering(n_clusters=clusters)
#     yhat_AC = AC.fit_predict(PCA_ds)
#     PCA_ds["Clusters"] = yhat_AC
#     data["Clusters"] = yhat_AC

#     pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
#     f, ax = plt.subplots(figsize=(5,5))

#     pl = sns.countplot(x=data["Clusters"], palette=pal)
#     ax.figure.savefig(buffer, format="png")
#     buffer.seek(0)
#     image_memory = base64.b64encode(buffer.getvalue())
#     doc = image_memory.decode('utf-8')

#     f, ax = plt.subplots(figsize=(5,5))
#     pl = sns.scatterplot(data=data, x=data["Spent"], y=data["Income"], hue=data["Clusters"], palette=pal)
#     plt.legend()
#     ax.figure.savefig(buffer, format="png")
#     buffer.seek(0)
#     image_memory = base64.b64encode(buffer.getvalue())
#     profile = image_memory.decode('utf-8')
    

#     data["Total_Promos"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data[
#         "AcceptedCmp5"]
#     f, ax = plt.subplots(figsize=(5,5))
#     pl = sns.countplot(x=data["Total_Promos"], hue=data["Clusters"], palette=pal)
#     pl.set_xlabel("Number Of Total Accepted Promotions")
#     ax.figure.savefig(buffer, format="png")
#     buffer.seek(0)
#     image_memory = base64.b64encode(buffer.getvalue())
#     promote = image_memory.decode('utf-8')

#     f, ax = plt.subplots(figsize=(5,5))
#     pl = sns.boxenplot(y=data["NumDealsPurchases"], x=data["Clusters"], palette=pal)
#     pl.set_title("Number of Deals Purchased")
#     ax.figure.savefig(buffer, format="png")
#     buffer.seek(0)
#     image_memory = base64.b64encode(buffer.getvalue())
#     deals = image_memory.decode('utf-8')

#     return render_template('output2.html', doc=doc, profile=profile, promote=promote, deals=deals)

@app.route('/r')
def r(): 
  return render_template('r.html')

@app.route("/comp_result", methods = ['POST'])
def comp_result():
    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    dataset.save(os.path.join(UPLOAD_FOLDER, filename))
    path = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(path)

    k = int(request.form['k'])
    test = int(request.form['test'])
    target = request.form['target']
    depth = int(request.form['depth'])

    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.fillna(0)
    df.dropna(inplace=True)
    l = LabelEncoder()

    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))

    for i in object_cols_float:
        df[i] = l.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)

    to_drop = []
    f, ax = plt.subplots(figsize=(12,10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')

    cor_target = abs(cor[target])
    relevant_features = cor_target[cor_target < 0.5]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)
    Y = df[target]
    X = df.drop(target, axis=1)

    # Bifurcating the training and testing data among the Dependent variable and Independent variable
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, train_size=(1-test), random_state=0)
    X_train.dropna(inplace=True)
    Y_train.dropna(inplace=True)
    X_test.dropna(inplace=True)
    Y_test.dropna(inplace=True)

    # Random Forest Training
    RF = RandomForestClassifier()
    RF.fit(X_train, Y_train)
    acc_RF = RF.score(X_test, Y_test) * 100
    Y_predict_RF = RF.predict(X_test)
    y_predict_RF = RF.predict_proba(X_test)[:, 1]
    cm_Rf = confusion_matrix(Y_test, Y_predict_RF)
    f_s_RF = f1_score(Y_test, Y_predict_RF, average='weighted')

    # KNN Training
    KNN = Kn(n_neighbors=k)
    KNN.fit(X_train, Y_train)
    acc_KNN = KNN.score(X_test, Y_test) * 100
    Y_predict_KNN = KNN.predict(X_test)
    y_predict_KNN = KNN.predict_proba(X_test)[:, 1]
    cm_KNN = confusion_matrix(Y_test, Y_predict_KNN)
    f_s_KNN = f1_score(Y_test, Y_predict_KNN, average='weighted')

    # SVM Training
    svc = SVC(probability=True)
    svc.fit(X_train, Y_train)
    acc_SVM = svc.score(X_test, Y_test) * 100
    Y_predict_SVM = svc.predict(X_test)
    y_predict_SVM = svc.predict_proba(X_test)[:, 1]
    cm_SVM = confusion_matrix(Y_test, Y_predict_SVM)
    f_s_SVM = f1_score(Y_test, Y_predict_SVM, average='weighted')

    # Decision Tree Training
    DT = tree.DecisionTreeClassifier(max_depth=depth)
    DT.fit(X_train, Y_train)
    acc_DT = DT.score(X_test, Y_test) * 100
    Y_predict_DT = DT.predict(X_test)
    y_predict_DT = DT.predict_proba(X_test)[:, 1]
    cm_DT = confusion_matrix(Y_test, Y_predict_DT)
    f_s_DT = f1_score(Y_test, Y_predict_DT, average='weighted')

    # Logistic Regression Training
    logr = LogisticRegression()
    logr.fit(X_train, Y_train)
    acc_LR = logr.score(X_test, Y_test) * 100
    Y_predict_LR = logr.predict(X_test)
    y_predict_LR = logr.predict_proba(X_test)[:, 1]
    cm_LR = confusion_matrix(Y_test, Y_predict_LR)
    f_s_LR = f1_score(Y_test, Y_predict_LR, average='weighted')


    def add_labels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i]._round(2), y[i].__round_(2))


    accu = {'Random Forest': acc_RF, 'KNN': acc_KNN, 'Decision Tree': acc_DT, 'SVM': acc_SVM, 'Logistic Regression': acc_LR}
    Algorithm = list(accu.keys())
    Accuracy = list(accu.values())
    f, ax = plt.subplots(figsize=(5,5))
    plt.bar(Algorithm, Accuracy, width=0.4)
    add_labels(Algorithm, Accuracy)
    # plt.title("Accuracy Comparisons of Different Algorithms ")
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    acc = image_memory.decode('utf-8')
    # plt.show()

    score = {'Random Forest': f_s_RF, 'KNN': f_s_KNN, 'Decision Tree': f_s_DT, 'SVM': f_s_SVM,
            'Logistic Regression': f_s_LR}
    Algorithm = list(score.keys())
    F1_Score = list(score.values())
    f, ax = plt.subplots(figsize=(12, 10))
    plt.bar(Algorithm, F1_Score, width=0.4)
    add_labels(Algorithm, F1_Score)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    f1 = image_memory.decode('utf-8')
    
    fpr1, tpr1, threshold = roc_curve(Y_test, y_predict_DT, pos_label=1)
    fpr2, tpr2, threshold = roc_curve(Y_test, y_predict_RF, pos_label=1)
    fpr3, tpr3, threshold = roc_curve(Y_test, y_predict_LR, pos_label=1)
    fpr4, tpr4, threshold = roc_curve(Y_test, y_predict_SVM, pos_label=1)
    fpr5, tpr5, threshold = roc_curve(Y_test, y_predict_KNN, pos_label=1)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)
    roc_auc5 = auc(fpr5, tpr5)

    f, ax = plt.subplots(figsize=(12,10))
    plt.plot(fpr1, tpr1, label='AUC Decision Tree = %0.2f' % roc_auc1)
    plt.plot(fpr2, tpr2, label='AUC Random Forest = %0.2f' % roc_auc2)
    plt.plot(fpr3, tpr3, label='AUC Logistic Regression = %0.2f' % roc_auc3)
    plt.plot(fpr4, tpr4, label='AUC SVM = %0.2f' % roc_auc4)
    plt.plot(fpr5, tpr5, label='AUC KNN = %0.2f' % roc_auc5)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    roc = image_memory.decode('utf-8')
    # plt.show()

    return render_template('output1.html',corr = corr, roc = roc, acc = acc, f1 = f1)


@app.route('/cm')
def cm(): 
  return render_template('cm.html')

@app.route('/vtrons')
def vtrons(): 
  return render_template('vtrons.html')

@app.route("/goggles_result", methods = ['POST'])
def goggles_result():
    a = int(request.form['k'])
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        overlay = cv2.imread('static/Glasses/glass{}.png'.format(a), cv2.IMREAD_UNCHANGED)
            
        _, frame = cap.read()
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_scale)
        for (x, y, w, h) in faces:
            #cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
            overlay_resize = cv2.resize(overlay,(w,int(h*0.8)))
            frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    return render_template('vtrons.html')
    
@app.route('/vtronf')
def vtronf(): 
  return render_template('vtronf.html')

@app.route("/vtronfoutput", methods=['POST'])
def vtronfoutput(): 
    import mediapipe as mp
    import cv2

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic 

    cap = cv2.VideoCapture(0)

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue  # skip frames if not capturing

            # Convert the frame color from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with holistic model
            results = holistic.process(image)

            # Convert back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Face landmarks using FACEMESH_TESSELATION
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )

            # 2. Right hand landmarks
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )

            # 3. Left hand landmarks
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )

            # 4. Pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # Display the webcam feed
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('vtronf.html')


@app.route('/vtronc')
def vtronc(): 
  return render_template('vtronc.html')

@app.route('/vtroncoutput', methods=['POST'])
def vtroncoutput(): 
    import os
    import cv2
    import cvzone
    from flask import request, render_template

    x = int(request.form['x'])
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    # Use dynamic project root path to avoid hardcoding issues
    shirtFolderPath = os.path.join(app.root_path, 'static', 'clothes', 'Shirts')
    listShirts = os.listdir(shirtFolderPath)

    fixedRatio = 262 / 190  # widthOfShirt / width between landmarks
    shirtRatioHeightWidth = 581 / 440

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

        if lmList:
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]

            # Load selected shirt image
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[x]), cv2.IMREAD_UNCHANGED)

            # Use abs to avoid negative width and check before resizing
            widthOfShirt = abs(lm11[0] - lm12[0]) * fixedRatio
            widthOfShirt = int(widthOfShirt)

            if widthOfShirt > 0:
                imgShirt = cv2.resize(
                    imgShirt,
                    (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth))
                )

                currentScale = abs(lm11[0] - lm12[0]) / 190
                offset = int(44 * currentScale), int(48 * currentScale)

                try:
                    img = cvzone.overlayPNG(
                        img,
                        imgShirt,
                        (lm12[0] - offset[0], lm12[1] - offset[1])
                    )
                except Exception as e:
                    print(f"Overlay error: {e}")
            else:
                print("Invalid width detected. Skipping this frame.")

        # Show webcam feed with overlaid shirt
        cv2.imshow("Virtual Try-On", img)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('vtronc.html')



@app.route('/vtronm')
def vtronm(): 
  return render_template('vtronm.html')



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

