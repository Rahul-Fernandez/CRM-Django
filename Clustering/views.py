
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.conf.urls.static import static
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import base64
from io import BytesIO
from django.views.decorators.csrf import csrf_exempt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


@csrf_exempt
def predict(request):
         #return render(request, 'landing.html')
         day = request.GET['days']
         days = int(day)
         product = request.GET['product']
        #  return render('statdash.html')
         print(product)
         result = [[609.7893 , 611.13586, 611.16296, 610.7131 , 457.3406 , 611.0643 ],
         [609.24994, 610.55505, 610.6001 , 610.1488 , 460.67886, 610.4668 ],
         [608.6633 , 609.90137, 609.97473, 609.527  , 464.12503, 609.82574],
         [608.23724, 609.42065, 609.51587, 609.07086, 467.14478, 609.35236],
         [607.80176, 608.9314 , 609.0485 , 608.6059 , 470.15298, 608.87244],
         [607.4653 , 608.5488 , 608.6839 , 608.2431 , 472.89084, 608.4963 ],
         [607.13904, 608.17834, 608.3308 , 607.89154, 475.56973, 608.1329 ],
         [606.8771 , 607.8771 , 608.0445 , 607.6063 , 478.06482, 607.837  ],
         [606.63214, 607.5949 , 607.77655, 607.3391 , 480.48553, 607.5604 ],
         [606.43085, 607.35986, 607.554  , 607.11694, 482.77255, 607.3299 ],
         [606.2476 , 607.1449 , 607.35077, 606.9138 , 484.9838 , 607.1193 ],
         [606.09515, 606.96313, 607.1796 , 606.74255, 487.092  , 606.9414 ],
         [605.95935, 606.7998 , 607.02625, 606.5887 , 489.1286 , 606.78174],
         [605.84595, 606.6608 , 606.8963 , 606.45807, 491.08176, 606.6459 ],
         [605.7468 , 606.5375 , 606.7815 , 606.3424 , 492.96924, 606.52563],
         [605.6643 , 606.43243, 606.68414, 606.244  , 494.78665, 606.4232 ],
         [605.5935 , 606.3404 , 606.5995 , 606.158  , 496.54434, 606.3337 ],
         [605.5353 , 606.26196, 606.5279 , 606.0851 , 498.24142, 606.25757],
         [605.4865 , 606.19415, 606.46643, 606.0221 , 499.88428, 606.1919 ],
         [605.447  , 606.1367 , 606.41504, 605.96906, 501.47354, 606.13654],
         [605.4151 , 606.08765, 606.3716 , 605.9238 , 503.01337, 606.08936],
         [605.3902 , 606.0465 , 606.33575, 605.88617, 504.50497, 606.05   ],
         [605.371  , 606.01184, 606.30615, 605.8545 , 505.95123, 606.017  ],
         [605.357  , 605.9831 , 606.28217, 605.82855, 507.35352, 605.9898 ],
         [605.3473 , 605.95935, 606.2628 , 605.8071 , 508.714  , 605.9676 ]]
         attributes = {"Shirt": 1, "T-Shirt": 2, "Pants": 3, "Casuals": 4, "Shoes": 5, "Accessories": 6}
         out_number = attributes[product]
         actual_results = []
         my_array = [i for i in range(1, days+1)]
         for i in range(days):
            actual_results.append(result[i][out_number-1])
         actual_results
         context = {
             'result' : actual_results,
              'days' : my_array

         }
         return render(request,'predict.html',context)

def preprocessing(request):
    # request.session.clear()
    if bool(request.FILES.get('document', False)) == True:
        uploaded_file = request.FILES['document']
        name = uploaded_file.name
        request.session['name'] = name
        df = pd.read_csv(uploaded_file)
        dataFrame = df.to_json()
        request.session['df'] = dataFrame
        
        rows = len(df.index)
        request.session['rows'] = rows
        header = df.axes[1].values.tolist()
        request.session['header'] = header
        
        attributes = len(header)
        types = []
        maxs = []
        mins = []
        means = []
        # statistic attribut
        for i in range(len(header)):
            types.append(df[header[i]].dtypes)
            if df[header[i]].dtypes != 'object':
                maxs.append(df[header[i]].max())
                mins.append(df[header[i]].min())
                means.append(round(df[header[i]].mean(),2))
            else:
                maxs.append(0)
                mins.append(0)
                means.append(0)

        zipped_data = zip(header, types, maxs, mins, means)
        print(maxs)
        datas = df.values.tolist()
        data ={  
                "header": header,
                "headers": json.dumps(header),
                "name": name,
                "attributes": attributes,
                "rows": rows,
                "zipped_data": zipped_data,
                'df': datas,
                "type": types,
                "maxs": maxs,
                "mins": mins,
                "means": means,
            }
    else:
        name = 'None'
        attributes = 'None'
        rows = 'None'
        data ={
                "name": name,
                "attributes": attributes,
                "rows": rows,
            }
    return render(request, 'index.html', data) 

def checker_page(request):
    if request.POST:
        drop_header = request.POST.getlist('drop_header')
        print(drop_header)
        for head in drop_header:
            print(head)
        request.session['drop'] = drop_header
        method = request.POST.get('selected_method')
        if method == '1':
            return redirect('classification')
        elif method == '2':
            return redirect('clustering')
        else: 
            return redirect('preprocessing')
    else:
        return render(request, 'index.html')

def chooseMethod(request):
    if request.method == 'POST':
        method = request.POST.get('method')
        print('method di session : ', method)
        request.session['method'] = method
    return redirect('classification')

def classification(request):
    rows = request.session['rows']
    name = request.session['name']
    headers = request.session['header']
    print('header : ', headers)
    df = request.session['df']
    df = pd.read_json(df)
    print(df)
    if request.session:
        features = request.session['drop']
        print('features : ', features)
        method, k, graph, reportNB, reportKNN, options, crossValue, splitValue, crossValues, splitValues, outputs =  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        if request.session:
            method = request.session['method']
            print('method di class : ', method)
            if method == '1':
                nameMethod = 'K-Nearest Neighbors'
                k = request.POST.get('knn')
                print(k)
                if request.POST.get('validation'):
                    options = request.POST['validation']
                    print('options selected : ', options)
                    if options == '1':
                        splitValue = request.POST['splitValue']
                        print('split value : ', splitValue)
                        reportKNN, graph = knn(df, features, options, splitValue, 0, k)
                    elif options == '2':
                        crossValue = request.POST['crossValue']
                        print('cross value : ', crossValue)
                        reportKNN, graph = knn(df, features, options, 0, crossValue,  k)
            elif method == '2':
                nameMethod = 'Naive Bayes'
                outputs = request.POST.get('output')
                if request.POST.get('validation'):
                    options = request.POST['validation']
                    print('options selected : ', options)
                    if options == '1':
                        splitValue = request.POST['splitValue']
                        print('split value : ', splitValue)
                        reportNB, graph = naiveBayes(df, features, options, splitValue, 0, outputs)
                    elif options == '2':
                        crossValue = request.POST['crossValue']
                        print('cross value : ', crossValue)
                        reportNB, graph = naiveBayes(df, features, options, 0, crossValue, outputs)
            if crossValue:
                request.session['cross'] = crossValue
                crossValues = request.session['cross']
            elif splitValue:
                request.session['split'] = splitValue
                splitValues = request.session['split']

        data = {
            "headers": headers,
            "method": method,
            "naiveBayes": round((reportNB*100),2),
            "knn": round((reportKNN*100),2),
            "k": k,
            "name": name,
            "rows": rows,
            "nameMethod": nameMethod,
            "attributes": features,
            "mode": options,
            "output": outputs,
            "splitValue": splitValues,
            "crossValue": crossValues,
            "confusion": graph,
        }
    else:
        return redirect('preprocessing')
    return render(request, 'classification.html', data)

def clustering(request):
    rows = request.session['rows']
    name = request.session['name']
    df = request.session['df']
    df = pd.read_json(df)
    print(df)
    features = request.session['drop']
    print(features)
    nilai_x = features[0]
    nilai_y = features[1]
    if request.method == 'POST' and request.POST['nilai_k']:
        k = request.POST['nilai_k']
        nilai_k = int(k)

        x_array = np.array(df.iloc[:, 3:5])

        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x_array)

        # Menentukan dan mengkonfigurasi fungsi kmeans
        kmeans = KMeans(n_clusters = nilai_k)
        # Menentukan kluster dari data
        kmeans.fit(x_scaled)

        # Menambahkan kolom "kluster" dalam data frame
        df['cluster'] = kmeans.labels_
        cluster = df['cluster'].value_counts()
        clusters = cluster.to_dict()
        sort_cluster = []
        label = []
        for i in sorted(clusters):
            sort_cluster.append(clusters[i])
            label.append(i)
        
        fig, ax = plt.subplots()
        sct = ax.scatter(x_scaled[:,1], x_scaled[:,0], s = 200, c = df.cluster)
        legend1 = ax.legend(*sct.legend_elements(),loc="lower left", title="Clusters")
        ax.add_artist(legend1)
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:,1], centers[:,0], c='red', s=200)
        plt.title("Clustering K-Means Results")
        plt.xlabel(nilai_x)
        plt.ylabel(nilai_y)
        graph = get_graph()

        if name:
            data = {
                "name": name,
                "clusters": sort_cluster,
                "rows": rows,
                "features": features,
                "label": label,
                "chart": graph,
            }
    else:
        data = {
            "name": '',
        }

    return render(request, 'clustering.html', data) 

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def naiveBayes(df, features, options, size, fold, outputs):
    # Variabel independen
    fitur = features
    x = df[fitur]
    # Variabel dependen
    y = df[outputs]
    # mengubah nilai fitur menjadi rentang 0 - 1
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    from sklearn.naive_bayes import GaussianNB
    # Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive Bayes
    modelnb = GaussianNB()
    train = options

    split = (int(size))/100
    cross = int(fold)

    if train == '1':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split)
        # Memasukkan data training pada fungsi klasifikasi Naive Bayes
        nbtrain = modelnb.fit(x_train, y_train)
        # Menentukan hasil prediksi dari x_test
        y_pred = nbtrain.predict(x_test)
        ytest = np.array(y_test)
        y_test = ytest.flatten()
        report = metrics.accuracy_score(y_test, y_pred) #score prediksi

        f, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        graph = get_graph()

        classification_report(y_test, y_pred)
        print(report)
        return report, graph

    elif train == '2':
        # k - fold cross validation
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(modelnb, x, y, cv=cross)
        report = metrics.accuracy_score(y, y_pred)  #score prediksi
        
        f, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt=".0f", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        graph = get_graph()

        classification_report(y, y_pred)
        print(report)
        return report, graph

def knn(df, features, options, size, fold, kValue):
    fitur = features
    x = df[fitur]

    y = df.iloc[:,-1:]

    # mengubah nilai fitur menjadi rentang 0 - 1
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    k = int(kValue)
    split = (int(size))/100
    cross = int(fold)

    # pemanggilan library KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    train = options

    if train == '1':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split)

        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        report = metrics.accuracy_score(y_test, y_pred) #score prediksi

        f, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        graph = get_graph()

        classification_report(y_test, y_pred)
        print(report)
        return report, graph

    elif train == '2':
        # k - fold cross validation
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(knn, x, y, cv=cross)
        report = metrics.accuracy_score(y, y_pred)  #score prediksi
        
        f, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt=".0f", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        graph = get_graph()

        classification_report(y, y_pred)
        print(report)
        return report, graph
