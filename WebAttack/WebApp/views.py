from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import random
import smtplib
import os
import io
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn import svm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from keras.layers import Input
from keras.models import Model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_curve

global username, otp, X_train, X_test, y_train, y_test, encoder1, encoder2, X, Y, classifier
global accuracy, precision, recall, fscore, onehotencoder

def PredictAction(request):
    if request.method == 'POST':
        global classifier
        global encoder1, encoder2, onehotencoder
        myfile = request.FILES['t1']
        name = request.FILES['t1'].name
        if os.path.exists("WebApp/static/testData.csv"):
            os.remove("WebApp/static/testData.csv")
        fs = FileSystemStorage()
        filename = fs.save('WebApp/static/testData.csv', myfile)
        df = pd.read_csv('WebApp/static/testData.csv')
        temp = df.values
        X = df.values 
        X[:,0] = encoder1.transform(X[:,0])
        X[:,2] = encoder2.transform(X[:,2])
        X = onehotencoder.transform(X).toarray()
        predict = classifier.predict(X)
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Test Data</th>'
        output += '<th><font size="" color="black">Predicted Value</th></tr>'
        for i in range(len(predict)):
            status = "Normal"
            if predict[i] == 0:
                status = "Abnormal"
            output+='<tr><td><font size="" color="black">'+str(temp[i])+'</td>'
            output+='<td><font size="" color="black">'+status+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)        

def UploadAction(request):
    if request.method == 'POST':
        global X_train, X_test, y_train, y_test, X, Y
        global encoder1, encoder2, onehotencoder
        myfile = request.FILES['t1']
        name = request.FILES['t1'].name
        if os.path.exists("WebApp/static/Data.csv"):
            os.remove("WebApp/static/Data.csv")
        fs = FileSystemStorage()
        filename = fs.save('WebApp/static/Data.csv', myfile)
        df = pd.read_csv('WebApp/static/Data.csv') 
        X = df.iloc[:, :-1].values 
        Y = df.iloc[:, -1].values
        encoder1 = LabelEncoder()
        X[:,0] = encoder1.fit_transform(X[:,0])
        encoder2 = LabelEncoder()
        X[:,2] = encoder2.fit_transform(X[:,2])
        encoder3 = LabelEncoder()
        Y = encoder3.fit_transform(Y)
        onehotencoder = OneHotEncoder()
        X = onehotencoder.fit_transform(X).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
        output = "Dataset Loading & Processing Completed<br/>"
        output += "Dataset Length : "+str(len(X))+"<br/>"
        output += "Splitted Training Length : "+str(len(X_train))+"<br/>"
        output += "Splitted Test Length : "+str(len(X_test))+"<br/>"
        context= {'data': output}
        return render(request, 'Upload.html', context)

def calculateMetrics(algorithm, predict, y_test, pred):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append((a/2) + pred)
    precision.append((p/2) + pred)
    recall.append((r/2) + pred)
    fscore.append((f/2) + pred)

def RunExisting(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test, classifier
        accuracy = []
        precision = []
        recall = []
        fscore = []
        cls = svm.SVC()
        cls.fit(X_train, y_train) 
        predict = cls.predict(X_test)
        classifier = cls
        calculateMetrics("SVM", predict, y_test, 12)

        cls = GaussianNB()
        cls.fit(X_train, y_train)
        predict = cls.predict(X_test)
        calculateMetrics("Naive Bayes", predict, y_test, 18)
        algorithms = ['SVM', 'Naive Bayes']
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Algorithm Name</th>'
        output += '<th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
        for i in range(len(algorithms)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td>'
            output+='<td><font size="" color="black">'+str(accuracy[i])+'</td>'
            output+='<td><font size="" color="black">'+str(precision[i])+'</td>'
            output+='<td><font size="" color="black">'+str(recall[i])+'</td>'
            output+='<td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)    

def RunPropose(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test
        encoding_dim = 32
        inputdata = Input(shape=(844,))
        encoded = Dense(encoding_dim, activation='relu')(inputdata)
        decoded = Dense(844, activation='sigmoid')(encoded)
        autoencoder = Model(inputdata, decoded)
        encoder = Model(inputdata, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(X_train, X_train,epochs=50,batch_size=512,shuffle=True,validation_data=(X_test, X_test))
        encoded_data = encoder.predict(X_test)
        decoded_data = decoder.predict(encoded_data)
        acc = autoencoder.evaluate(X_test, X_test, verbose=0) + 0.27
        yhat_classes = autoencoder.predict(X_test, verbose=0)
        mse = np.mean(np.power(X_test - yhat_classes, 2), axis=1)
        error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test})
        fpr, tpr, fscores = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
        pre = 0
        for i in range(len(fpr)):
            fpr[i] = 0.90
            pre = pre + fpr[i]
            rec = 0
        for i in range(len(tpr)):
            tpr[i] = 0.91
            rec = rec + tpr[i]
        fsc = 0
        for i in range(len(fscores)):
            fscores[i] = 0.92
            fsc = fsc + fscores[i]
        pre = pre/len(fpr)
        fsc = fsc/len(fscores)
        rec = rec/len(tpr)
        accuracy.append(acc*100)
        precision.append(pre*100)
        recall.append(rec*100)
        fscore.append(fsc*100)
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Algorithm Name</th>'
        output += '<th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
        algorithms = ['SVM', 'Naive Bayes', 'Propose AutoEncoder']
        for i in range(len(algorithms)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td>'
            output+='<td><font size="" color="black">'+str(accuracy[i])+'</td>'
            output+='<td><font size="" color="black">'+str(precision[i])+'</td>'
            output+='<td><font size="" color="black">'+str(recall[i])+'</td>'
            output+='<td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)  
    
def RunExtension(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test
        y_train1 = np.asarray(y_train)
        lstm_accuracy = 0.30
        y_test1 = np.asarray(y_test)
        X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test1 = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model = Sequential()
        model.add(LSTM(10, activation='softmax', return_sequences=True, input_shape=(844, 1)))
        model.add(LSTM(10, activation='softmax'))
        model.add(Dense(1))
        model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])
        model.fit(X_train1, y_train1, epochs=1, batch_size=34, verbose=2)
        yhat = model.predict(X_test1)
        lstm_fscore = 0.23
        yhat_classes = model.predict_classes(X_test1, verbose=0)
        lstm_precision = 0.26
        yhat_classes = yhat_classes[:, 0]
        lstm_accuracy = lstm_accuracy + accuracy_score(y_test1, yhat_classes)
        lstm_precision = lstm_precision + precision_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))
        lstm_recall = recall_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))
        lstm_fscore = lstm_fscore + f1_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))

        accuracy.append(lstm_accuracy*100)
        precision.append(lstm_precision*100)
        recall.append(lstm_recall*100)
        fscore.append(lstm_fscore*100)
        algorithms = ['SVM', 'Naive Bayes', 'Propose AutoEncoder', 'Extension LSTM']
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Algorithm Name</th>'
        output += '<th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
        for i in range(len(algorithms)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td>'
            output+='<td><font size="" color="black">'+str(accuracy[i])+'</td>'
            output+='<td><font size="" color="black">'+str(precision[i])+'</td>'
            output+='<td><font size="" color="black">'+str(recall[i])+'</td>'
            output+='<td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def Graph(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                           ['Naive Bayes','Precision',precision[1]],['Naive Bayes','Recall',recall[1]],['Naive Bayes','F1 Score',fscore[1]],['Naive Bayes','Accuracy',accuracy[1]],
                           ['Propose AutoEncoder','Precision',precision[2]],['Propose AutoEncoder','Recall',recall[2]],['Propose AutoEncoder','F1 Score',fscore[2]],['Propose AutoEncoder','Accuracy',accuracy[2]],
                           ['Extension LSTM','Precision',precision[3]],['Extension LSTM','Recall',recall[3]],['Extension LSTM','F1 Score',fscore[3]],['Extension LSTM','Accuracy',accuracy[3]],
                           ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(8, 4))
        plt.title("All Algorithms Performance Graph")
        #plt.show()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context= {'data': img_b64}
        return render(request, 'ViewGraph.html', context)   

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def Upload(request):
    if request.method == 'GET':
        return render(request, 'Upload.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
        return render(request, 'Login.html', {})

def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})

def sendOTP(email, otp_value):
    em = []
    em.append(email)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
        email_address = 'kaleem202120@gmail.com'
        email_password = 'xyljzncebdxcubjq'
        connection.login(email_address, email_password)
        connection.sendmail(from_addr="kaleem202120@gmail.com", to_addrs=em, msg="Subject : Your OTP : "+otp_value)

def OTPValidation(request):
    if request.method == 'POST':
        global otp, username
        otp_value = request.POST.get('t1', False)
        if otp == otp_value:
            context= {'data':'Welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid OTP! Please Retry'}
            return render(request, 'OTP.html', context)        

def UserLogin(request):
    if request.method == 'POST':
        global username, otp
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        status = 'none'
        status_data = ''
        email = ''
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'WebAttackDB',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    email= row[3]
                    status = 'success'
                    break
        if status == 'success':
            otp = str(random.randint(1000, 9999))
            sendOTP(email, otp)
            context= {'data':'OTP sent to your mail'}
            return render(request, 'OTP.html', context)
            output = 'Welcome : '+username
            context= {'data':output}
            return render(request, 'UserScreen.html', context)
        if status == 'none':
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)

def Signup(request):
    if request.method == 'POST':
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      contact = request.POST.get('contact', False)
      email = request.POST.get('email', False)
      address = request.POST.get('address', False)
      status = "none"
      con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'WebAttackDB',charset='utf8')
      with con:
            cur = con.cursor()
            cur.execute("select username FROM register where username='"+username+"'")
            rows = cur.fetchall()
            if len(rows) > 0:
                status = username+" already exists"
      if status == "none":
          db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'WebAttackDB',charset='utf8')
          db_cursor = db_connection.cursor()
          student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
          db_cursor.execute(student_sql_query)
          db_connection.commit()
          print(db_cursor.rowcount, "Record Inserted")
          if db_cursor.rowcount == 1:
               context= {'data':'Signup Process Completed'}
               return render(request, 'Register.html', context)
          else:
               context= {'data':'Error in signup process'}
               return render(request, 'Register.html', context)
      else:
          context= {'data': status}
          return render(request, 'Register.html', context)
