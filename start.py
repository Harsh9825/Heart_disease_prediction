import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/Harsh Singh/OneDrive/Documents/Jupyer/trained_model.sav', 'rb'))

input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)
input_data_as_numpy_array= np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print('The persion does not have a heart disease')
else:
    print('The persion have a heart disease')