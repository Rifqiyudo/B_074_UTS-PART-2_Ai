#Rifqi yudo dewantoro_21091397074
#Multiple perceptron / Neuron batch and multiple layer 2

#inisialisasi numpy
import numpy as np

# inisialisasi variabel
# memasukan nilai variabel layer feature 10 dengan batch sejumlah 6 
inputs = [
    [5.0, 1.3, 2.3, 2.6, 3.1, 3.8, 4.6, 4.9, 5.2, 5.2],
    [1.3, 1.8, 1.2, 2.4, 4.2, 2.4, 3.2, 4.2, 5.1, 5.4],
    [3.1, 20.5, 10.0, 25.5, 35.0, 31.5, 18.0, 50.5, 51.0, 57.5],
    [2.8, 2.8, 2.5, 4.8, 3.2, 3.1, 2.6, 4.9, 5.2, 5.1],
    [2.2, 5.4, 7.2, 7.4, 8.2, 8.4, 9.2, 9.4, 10.2, 10.4],
    [16.5, 18.4, 15.2, 16.4, 19.2, 12.4, 17.2, 12.4, 22.2, 11.4]
    ]

# memberikan nilai bobot pada variabel sesuai dengan jumlah input
# memasukan jumlah weight sesuai dengan jumlah neuron yaitu sejumlah 5  
weights1 = [
    [6.0, 4.8, 8.4, 2.5, 0.1, 3.5, 9.7, 4.5, 6.2, 15.5],
    [7.4, 9.7, 4.10, 2.84, 3.52, 38.4, 45.2, 4.4, 5.2, 5.4],
    [3.3, 6.1, 2.3, 10.9, 31.6, 3.82, 4.26, 4.8, 56.6, 55.8],
    [5.8, 4.3, 4.2, 7.8, 0.2, 7.4, 3.5, 0.7, 40.3, 71.1],
    [5.1, 13.7, 30.6, 42.7, 95.1, 12.3, 29.0, 40.7, 28.1, 93.11],
    ]

# inisialisasi biases pada layer1 sesuai dengan neuron yang ditentukan yaitu layer 1 = 5 neuron
biases1 =   [4.7, 2.8, 1.0, 9.6, 3.1]

# inisialisasi jumlah weight 2, weight layer 2 = neuron layer 1 yaitu 5
# memasukkan jumlah weight sesuai dengan neuron layer 2 yaitu 3 neuron
weights2 = [
    [15.3, 7.4, 5.9, 8.2, 12.6],
	[7.0, 1.2, 4.7, 7.4, 9.7],
	[2.1, 6.9, 3.7, 4.0, 3.4]]

# inisialisasi biases pada layer2 dengan neuron yang ditentukan yaitu 3			
biases2 =  [8.8, 4.6, 5.3]


# output
# menghitung layer1 dengan (inputs*weight1) dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# menghitung layer2 dengan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print output layer2
print(layer2_outputs) 