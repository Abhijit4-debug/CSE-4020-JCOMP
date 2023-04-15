import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt



model = Sequential()
model.add(Dense(20, input_dim=33, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='softmax')) #for multiclass classification
    #Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy']
                 )


df3=df3.drop(columns=['Unnamed: 0'],axis=1)
X1=df3.drop(columns=['label'],axis=1)
Y1 = df3['label']



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        model.set_weights(parameters)
        r = model.fit(X,Y,epochs=10,validation_data=(X1, Y1),batch_size=2048)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(X), {}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Testing ----------------- ")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X1, Y1, verbose=0)
        print("Eval accuracy : ", accuracy)
        y_pred=np.argmax(model.predict(np.array(X1)), axis=1)
        
        print("Mean Absolute Error - " , metrics.mean_absolute_error(Y1, y_pred))
        print("Mean Squared Error - " , metrics.mean_squared_error(Y1, y_pred))
        print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(Y1, y_pred)))
        print("R2 Score - " , metrics.explained_variance_score(Y1, y_pred)*100)
        print("Accuracy - ",accuracy_score(Y1,y_pred)*100)
        cf =confusion_matrix(Y1,y_pred)
        print("The classification metrics is  :  ",cf)
        print(classification_report(Y1, y_pred))
        cm = confusion_matrix(Y1, y_pred)
        # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
        cm_df = pd.DataFrame(cm,
                            index = ['Benign','Infilteration','Dos attacks-golden-eye','Dos attacks-slowloris'], 
                            
                            columns = ['Benign','Infilteration','Dos attacks-golden-eye','Dos attacks-slowloris'])
        #Plotting the confusion matrix
        plt.figure(figsize=(10,8))
        sns.heatmap(cm_df, annot=True,fmt=".1f",cmap="GnBu",linewidth=.5)
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()
        
        return loss, len(X1), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)
