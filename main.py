#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cluster import KMeans

#%%
os.getcwd()
#%%
name_of_activity=[
    'a01',  # 1
    'a02', # 2
    'a03', # 3
    'a04', # 4
    'a05' , # 5
    'a06', # 6
    'a07', # 7
    'a08', # 8
    'a09', # 9
    'a10', # 10
    'a11', # 11
    'a12', # 12
    'a13', # 13
    'a14', # 14
    'a15', # 15
    'a16', # 16
    'a17', # 17
    'a18', # 18
    'a19' # 19
    ]
normalized_values_list = []
patient_label_list = []
activity_label_list = []

for activity_folder in name_of_activity:
    
    os.chdir('C:/Users/arman/OneDrive/Desktop/ICT for Health/data')
    # going inside each activity folder
    os.chdir(activity_folder)
    print(activity_folder)
    patient_files = os.listdir()
    #Student ID
    ID=301000
    s=ID%8+1
    IDpatient=[patient_files[s-1]]
    
    for patient in IDpatient:
        
        # going into every patient folder
        os.chdir(patient)
        print(patient)
        segment_files = os.listdir()
        
        # getting all the segment txt files inside the patient folder
        print(segment_files)
        
        for filename in segment_files:
            
            # obtaining the 1170x1 vector, patient id, activity id from the text file.
            print('Doing {}'.format(filename))
            df = pd.read_csv('{}'.format(filename), header=None)

            # first step
            first_step = list(df.min())+list(df.max())+list(df.mean())+list(df.skew())+list(df.kurtosis())

            dft_matrix = np.fft.fft(df.to_numpy().T)
            abs_dft_matrix = np.absolute(dft_matrix)

            # second step and third step
            # If T is the total amount of time passed in signal that taking DFT
            # and k is the index
            # then the frequency at index k is:
            f_k = (2*math.pi)/5
            second_step =[]
            third_step = []
            for i in range(len(abs_dft_matrix)):
                positions = abs_dft_matrix[i].argsort()[-5:][::-1]
                second_step.append(list(abs_dft_matrix[i][positions]))
                third_step.append(list(positions*f_k))
            second_step = [item for sublist in second_step for item in sublist] # flattening the lists.
            third_step = [item for sublist in third_step for item in sublist]
            
            # fourth step   
            fourth_step = []
            autocorr_reqd = [0,4,9,14,19,24,29,34,39,44,49]

            for column in df.columns:
                mean = df[column].mean()
                for delta in range(len(df)):
                    if(delta in autocorr_reqd):
                        sum_of_products = 0
                        for i, row in enumerate(df[column], start = delta):
                            element_1 = row - mean
                            element_2 = df[column].iloc[len(df)-1-i] - mean
                            sum_of_products += element_1*element_2
                        rss = 1/(len(df)-delta)*sum_of_products 
                        fourth_step.append(rss)
            
            # finalizing
            final_representation = first_step + second_step + third_step + fourth_step
            arr = np.asarray(final_representation)
            normalized = (arr-min(arr))/(max(arr)-min(arr))
            
            normalized_values_list.append(list(normalized)) # a 2D list with 1140 lists insdie it, each has 1170 values.
            patient_label_list.append(patient) # a 1D list with 1140 patient ids.
            activity_label_list.append(activity_folder) # a 1D list iwth 1140 activity ids.
        
        os.chdir('C:/Users/arman/OneDrive/Desktop/ICT for Health/data/{}'.format(activity_folder))


#%%
actual = pd.DataFrame(normalized_values_list)


#%%
actual['patient'] = patient_label_list
actual['activity'] = activity_label_list

#%%
#The step below does LabelEncoding, which is converting all string categories ('a1', 'a2',.. and 'p1') 
# into numerical categories [0, 1, 2 .. 18] for activity , since scikit-learn understands numbers only.
X2 = actual.iloc[:,1170:1172]
X2 = X2.apply(LabelEncoder().fit_transform)
actual.drop(['patient', 'activity'], axis = 1, inplace = True)

#%%
#Merging the LabelEncoded df , X2 with actual and storing it in X_t
X_t = actual.join(X2)
X = X_t.iloc[:,0:1170]
y = X_t['activity']

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state = 301000)

#%%
model = KMeans(n_clusters=19, random_state=None, n_init="auto")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#%%
actNamesShort=[
    'sitting',  # 1
    'standing', # 2
    'lying.ba', # 3
    'lying.ri', # 4
    'asc.sta' , # 5
    'desc.sta', # 6
    'stand.elev', # 7
    'mov.elev', # 8
    'walk.park', # 9
    'walk.4.fl', # 10
    'walk.4.15', # 11
    'run.8', # 12
    'exer.step', # 13
    'exer.train', # 14
    'cycl.hor', # 15
    'cycl.ver', # 16
    'rowing', # 17
    'jumping', # 18
    'play.bb' # 19
    ]

#%%
a = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(18,14))
ax = sns.heatmap(a, linewidth=0.5, annot=True, fmt='g', annot_kws={"size": 14}, cmap='rocket_r')
plt.xlim([0,19])
plt.xticks(np.arange(0.5, 19, 1), labels=actNamesShort, rotation=90, ha='center')
plt.yticks(np.arange(0.5, 19, 1), labels=actNamesShort, rotation=0)
ax.xaxis.set_ticks_position('top')
plt.ylabel('Activities')
plt.xlabel('Activities')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=1200)
plt.show()

#%%
grouped = actual.groupby(np.arange(len(actual))//60).mean()
centroids = grouped.values
grouped2 = np.sqrt(actual.groupby(np.arange(len(actual))//60).var())
stdpoints = grouped2.values

#%%
centroids = np.delete(centroids, np.s_[-720:], axis=1)
stdpoints = np.delete(stdpoints, np.s_[-720:], axis=1)

#%%
Nsensors = 45
NAc = 19

#%%
d=np.zeros((NAc,NAc))
for i in range(NAc):
    for j in range(NAc):
        d[i,j]=np.linalg.norm(centroids[i]-centroids[j])

dd=d+np.eye(NAc)*1e6# remove zeros on the diagonal (distance of centroid from itself)
dmin=dd.min(axis=0)# find the minimum distance for each centroid
dmin*=10
dpoints=np.sqrt(np.sum(stdpoints**2,axis=1))
dpoints*=10


#%%
plt.figure()
plt.plot(dmin,label='minimum centroid distance')
plt.plot(dpoints,label='mean distance from points to centroid')
plt.grid()
plt.xticks(np.arange(NAc),actNamesShort,rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('cluster analysis.png', dpi=1200)