# coding: utf-8

# ## Drexel University
# ## CS-615: Deep Learning
# ## HW2
# ## John Obuch

############################# PART 2 ###################################

#import requirements
import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt

print("\nPART 2: Shallow Artificial Neural Networks")

#establish the directory where the files are located
trainDirectory = "yalefaces/"

#store all files in the /yalefaces directory into a list (NOTE: The README.txt file is an element in the list!)
lFileList = []
for fFileObj in os.walk(trainDirectory): 
    lFileList = fFileObj[2]
    break  

#define empty lists to store flattened image row vectors into as well as the class value labels
matrix = []
class_ = []

#obtain X matrix data and Y target vector (i.e. class values)
for file in lFileList:
    
    #if file is not the README.txt file
    if file != 'Readme.txt':
        
        #read in the image, cast to numpy array and resize the image to be dimensions 40x40, and flatten the image
        im = cv2.resize(np.array(Image.open(trainDirectory+file)), (40,40)).flatten()
        
        #reshape the flattened images and append the images to the matrix list
        im = np.reshape(im, (1,im.shape[0]))[0]
        
        #append each row vector to the matrix list
        matrix.append(im)
        
        #split the file and grab first element subject<id>
        subject_id = file.split(".")[0]
        
        #grab the last two values in the string (i.e. the <id> number) and cast it to an integer
        class_val = int(subject_id[-2:])
        
        #append the values to the class_ list
        class_.append(class_val)

#cast the matrix list and class list to a numpy arrays (matrix/vector)
X = np.array(matrix) 
Y= np.array(np.reshape(class_, (len(class_), 1))) 

#concatinate the y vector to the x matrix such that the last column in matrix A is the target values
A = np.concatenate((X, Y), axis=1)

#initialize random number generator
np.random.seed(1)

#create list to store sub matricies associated to each class into
class_matrix_L = []

#get the sub-matricies for each class
for c in set(A[:, -1]):
    mask = A[:, -1] == c
    A_c = A[mask]
    class_matrix_L.append(A_c)

#define empty lists to store the rows of train/test sub-matries into to built the training and testing matricies
train_temp = []
test_temp = []

#shuffle the sub-matrices and split them into 2/3 train and 1/3 test split to ensure equal class prior probabilities
for mat in class_matrix_L:
    np.random.shuffle(mat)
    cutoff = math.ceil((2/3)*mat.shape[0])
    train_ = mat[0:cutoff, :]
    test_ = mat[cutoff:, :]
    
    #for each row in the train and test sub-matricies, append each row to the train_temp & test_temp lists
    for row in train_:
        train_temp.append(row)    
    for row in test_:
        test_temp.append(row)
    
#cast the train and test lists to numpy array matricies
A_train = np.array(train_temp) #numpy.ndarray
A_test = np.array(test_temp)

#randomize the rows in the train and test Matrix groups
np.random.shuffle(A_train)
np.random.shuffle(A_test)

#establish train and test groups
X_train = np.array(A_train[:, 0:-1]) #train
Y_train = np.array(A_train[: , -1])
X_test = np.array(A_test[: , 0:-1]) #test
Y_test = np.array(A_test[: , -1])

#one-hot-encode the Y_train and Y_test vectors to be matricies where each column is representative of a class. 
#per Maryam (TA) ok to use sklearn onehot encoder!
enc = OneHotEncoder(categories='auto')
Y_train_onehot = enc.fit_transform(Y_train.reshape(-1, 1)).toarray()
Y_test_onehot = enc.fit_transform(Y_test.reshape(-1, 1)).toarray()

#compute mean and std of training data to identify if there is a zero in the std vector
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)  

#remove the features that corresponded to std of zero
for indx, i in enumerate(range(len(X_std_train))):
    if X_std_train[i] == 0:
        X_train = np.delete(X_train, indx, axis=1)
        X_test = np.delete(X_test, indx, axis=1)

#create vector of ones (i.e. create bias feature for both train/test groups)
bias_feature_train = np.ones(X_train.shape[0])
bias_feature_test = np.ones(X_test.shape[0])

#recompute mean and std of training data
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)

#standardize the data
X_stdz_train = (X_train - X_bar_train)/X_std_train 
X_stdz_test = (X_test - X_bar_train)/X_std_train

#Define standardized X matrix arrays (including bias feature)
X_stdz_train =np.column_stack([bias_feature_train, X_stdz_train[:, 0:X_stdz_train.shape[1]]])
X_stdz_test =np.column_stack([bias_feature_test, X_stdz_test[:, 0:X_stdz_test.shape[1]]]) 
# print(X_stdz_train)

#define the number of hidden layer nodes
N_hidden_nodes = 1000 

#initialize beta matrix (dimensions DxM where D is the number of input nodes and M is the number of hidden nodes)
beta = np.random.uniform(-1,1,(X_stdz_train.shape[1], N_hidden_nodes))

#initialize theta matrix (dimensions MxK where M is the number of hidden nodes and K is the number of class lables)
theta = np.random.uniform(-1,1,(N_hidden_nodes,Y_train_onehot.shape[1]))

#define the learning rate eta
eta = .01

# #since performing Batch Gradient Decent we need to compute eta/N where N here is the size (number of records) of X_train
eta_over_N = eta/X_stdz_train.shape[0]

#define the regularization term
lambda_ = .5

# #set the threshold values so we know when to stop
log_thresh = .001

# #set values of variables to compare to threshold values
chg_in_log = 1
prev_log = 1000

# #initialize the iteration count
iter_ = 0

#define empty lists to store iteration and cost history into
J_train = []
J_test = []

print("Training the data. Please wait...")

#perform Batch Gradient Decent until convergence criteria is met
while (chg_in_log > log_thresh): 
    
    #define the L2 norm regularization term
    L2 = np.linalg.norm(theta)
    
    ###### TEST DATA #######
    
    #compute the activation function output
    h_test = 1/(1 + np.exp(np.dot(-X_stdz_test, beta)))      
    
    #compute Y_hat_test prediction
    Y_hat_test = 1/(1 + np.exp(np.dot(-h_test, theta)))      
    
    #compute the error
    Y_err = (Y_test_onehot - Y_hat_test)

    #compute the current loss of the cost function for the testing data
    crnt_log_test = (1/X_stdz_test.shape[0])*(np.sum((Y_test_onehot) * np.log(Y_hat_test + 1e-5) + 
                                                      (1-Y_test_onehot) * np.log(1 - Y_hat_test + 1e-5)) - lambda_*L2)  
    #append the iterations and costs to the list
    J_test.append((iter_, crnt_log_test))
    
    ###### TRAIN DATA #######
    
    #compute the output of the hiden layer
    h_train = 1/(1 + np.exp(np.dot(-X_stdz_train, beta)))     
    
    #compute y_hat via the activation function
    Y_hat_train = 1/(1 + np.exp(np.dot(-h_train, theta)))      
    
    #compute the error (i.e. the residuals)
    Y_err = (Y_train_onehot - Y_hat_train)

    #update the current loss of the cost function for the trianing data
    crnt_log_train = (1/X_stdz_train.shape[0])*(np.sum((Y_train_onehot) * np.log(Y_hat_train + 1e-5) + 
                                                       (1-Y_train_onehot) * np.log(1-Y_hat_train + 1e-5)) - lambda_*L2) 
    
    #append the iterations and costs to the list
    J_train.append((iter_,crnt_log_train))
    
    #compute the gradient with respect to beta
    grad_beta = X_stdz_train.T @ (((Y_train_onehot - Y_hat_train) @ theta.T) * (h_train * (1 - h_train))) - lambda_*beta
    
    #perform the update of beta
    beta = beta + eta*grad_beta

    #compute the gradient with respect to theta
    grad_theta = np.dot(h_train.T, Y_err) - lambda_*theta   

    #perform the update of theta
    theta = theta + eta*grad_theta #eta is better vs eta_over_N
        
    #update the change in loss and the previous loss
    chg_in_log = abs((prev_log - crnt_log_train)/prev_log)
    prev_log = crnt_log_train

    #increment the iteration count
    iter_ += 1

#return the number iterations to the screen
print("TOTAL ITERATIONS")
print(iter_)

#compute the probabilities of the classification for the testing training sets
h_test = 1/(1 + np.exp(np.dot(-X_stdz_test, beta)))
Y_hat_test = 1/(1 + np.exp(np.dot(-h_test, theta))) #test

Y_h_train =1/(1 + np.exp(np.dot(-X_stdz_train,beta))) #train
Y_hat_train =1/(1 + np.exp(np.dot(-Y_h_train,theta)))

#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_test_pred = np.zeros([Y_hat_test.shape[0], Y_hat_test.shape[1]])  #Test
for i, row in enumerate(Y_hat_test):
    Y_hat_test_pred[i][np.argmax(row)] = 1
            
#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_train_pred = np.zeros([Y_hat_train.shape[0], Y_hat_train.shape[1]])  #Train
for i, row in enumerate(Y_hat_train):
    Y_hat_train_pred[i][np.argmax(row)] = 1

#keep track of iteration and cost for plots for both train and test
cost_train = []
iteration_train = []
for tup in J_train:
    cost_train.append(tup[1])
    iteration_train.append(tup[0]) #train
    
cost_test = []
iteration_test = []
for tup in J_test:
    cost_test.append(tup[1])
    iteration_test.append(tup[0]) #test

#confusion matrix train
print("\nTRAIN: CONFUSION MATRIX\n")
conf_train = Y_train_onehot.T @ Y_hat_train_pred
print(conf_train)
    
#confustion matrix test
print("\nTEST: CONFUSION MATRIX\n")
conf_test = Y_test_onehot.T @ Y_hat_test_pred    ##As long as you keep track of what axis
print(conf_test)

#compute the accuracy of the systems
print("\nTRAIN ACCURACY:")
acc_train = np.trace(conf_train)/np.sum(conf_train)
print(acc_train)

#return the test accuracy to the screen
print("\nTEST ACCURACY:")
acc_test = np.trace(conf_test)/np.sum(conf_test)
print(acc_test)

#plot the training cost outputs over the iterations
x_train = np.array(iteration_train)
y_train = np.array(cost_train)

fig, axes = plt.subplots()
axes.plot(x_train, y_train, 'b')
axes.set_title("Training Set Cost by Iteration Number")
axes.set_xlabel('Iteration')
axes.set_ylabel('Cost')
_ = plt.savefig("convergence_train_ann")

#plot the testing cost outputs over the iterations
x_test = np.array(iteration_test)
y_test = np.array(cost_test)

fig, axes = plt.subplots()
axes.plot(x_train, y_train, 'r')
axes.set_title("Testing Set Cost by Iteration Number")
axes.set_xlabel('Iteration')
axes.set_ylabel('Cost')
_ = plt.savefig("convergence_test_ann")

#######################################################################################################################################

####################### Part 3 ############################

#import requirements
import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt

print("\nPART 3: Multi-Layer ANN")

#establish the directory where the files are located
trainDirectory = "yalefaces/"

#store all files in the /yalefaces directory into a list (NOTE: The README.txt file is an element in the list!)
lFileList = []
for fFileObj in os.walk(trainDirectory): 
    lFileList = fFileObj[2]
    break  

#define empty lists to store flattened image row vectors into as well as the class value labels
matrix = []
class_ = []

#obtain X matrix data and Y target vector (i.e. class values)
for file in lFileList:
    
    #if file is not the README.txt file
    if file != 'Readme.txt':
        
        #read in the image, cast to numpy array and resize the image to be dimensions 40x40, and flatten the image
        im = cv2.resize(np.array(Image.open(trainDirectory+file)), (40,40)).flatten()
        
        #reshape the flattened images and append the images to the matrix list
        im = np.reshape(im, (1,im.shape[0]))[0]
        
        #append each row vector to the matrix list
        matrix.append(im)
        
        #split the file and grab first element subject<id>
        subject_id = file.split(".")[0]
        
        #grab the last two values in the string (i.e. the <id> number) and cast it to an integer
        class_val = int(subject_id[-2:])
        
        #append the values to the class_ list
        class_.append(class_val)

#cast the matrix list and class list to a numpy arrays (matrix/vector)
X = np.array(matrix) 
Y= np.array(np.reshape(class_, (len(class_), 1))) 

#concatinate the y vector to the x matrix such that the last column in matrix A is the target values
A = np.concatenate((X, Y), axis=1)

#initialize random number generator
np.random.seed(1)

#create list to store sub matricies associated to each class into
class_matrix_L = []

#get the sub-matricies for each class
for c in set(A[:, -1]):
    mask = A[:, -1] == c
    A_c = A[mask]
    class_matrix_L.append(A_c)

#define empty lists to store the rows of train/test sub-matries into to built the training and testing matricies
train_temp = []
test_temp = []

#shuffle the sub-matrices and split them into 2/3 train and 1/3 test split to ensure equal class prior probabilities
for mat in class_matrix_L:
    np.random.shuffle(mat)
    cutoff = math.ceil((2/3)*mat.shape[0])
    train_ = mat[0:cutoff, :]
    test_ = mat[cutoff:, :]
    
    #for each row in the train and test sub-matricies, append each row to the train_temp & test_temp lists
    for row in train_:
        train_temp.append(row)    
    for row in test_:
        test_temp.append(row)
    
#cast the train and test lists to numpy array matricies
A_train = np.array(train_temp) #numpy.ndarray
A_test = np.array(test_temp)

#randomize the rows in the train and test Matrix groups
np.random.shuffle(A_train)
np.random.shuffle(A_test)

#establish train and test groups
X_train = np.array(A_train[:, 0:-1]) #train
Y_train = np.array(A_train[: , -1])
X_test = np.array(A_test[: , 0:-1]) #test
Y_test = np.array(A_test[: , -1])

#one-hot-encode the Y_train and Y_test vectors to be matricies where each column is representative of a class. 
#per Maryam (TA) ok to use sklearn onehot encoder!
enc = OneHotEncoder(categories='auto')
Y_train_onehot = enc.fit_transform(Y_train.reshape(-1, 1)).toarray()
Y_test_onehot = enc.fit_transform(Y_test.reshape(-1, 1)).toarray()

#compute mean and std of training data to identify if there is a zero in the std vector
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)  

#remove the features that corresponded to std of zero
for indx, i in enumerate(range(len(X_std_train))):
    if X_std_train[i] == 0:
        X_train = np.delete(X_train, indx, axis=1)
        X_test = np.delete(X_test, indx, axis=1)

#create vector of ones (i.e. create bias feature for both train/test groups)
bias_feature_train = np.ones(X_train.shape[0])
bias_feature_test = np.ones(X_test.shape[0])

#recompute mean and std of training data
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)

#standardize the data
X_stdz_train = (X_train - X_bar_train)/X_std_train 
X_stdz_test = (X_test - X_bar_train)/X_std_train

#Define standardized X matrix arrays (including bias feature)
X_stdz_train =np.column_stack([bias_feature_train, X_stdz_train[:, 0:X_stdz_train.shape[1]]])
X_stdz_test =np.column_stack([bias_feature_test, X_stdz_test[:, 0:X_stdz_test.shape[1]]]) 

#establish a user input component and cast the input entires to a list/vector
layer_list = input("Enter the number of nodes per hidden layer seperated by a comma (e.g. 100,30,10,...,n): ")
input_vector = [int(i) for i in layer_list.split(',') if i.isdigit()]

#creat a list of matricies to initialize the weight params based on the provided input vector
weights = []
for i in range(len(input_vector)):
    if i == 0:
        w_mat = np.random.uniform(-1,1, (X_stdz_train.shape[1], input_vector[i]))
        weights.append(w_mat)
    else:
        w_mat = np.random.uniform(-1,1, (input_vector[i-1], input_vector[i]))
        weights.append(w_mat)

#initialize output theta param matrix (dimensions MxK where M is the number of hidden nodes and K is the number of class lables)
theta = np.random.uniform(-1,1,(input_vector[-1], Y_train_onehot.shape[1]))

#define the learning rate eta
eta = .01

# #since performing Batch Gradient Decent we need to compute eta/N where N here is the size (number of records) of X_train
eta_over_N = eta/X_stdz_train.shape[0]

#define the regularization term
lambda_ = .001 

# #set the threshold values so we know when to stop
log_thresh = .0001 

# #set values of variables to compare to threshold values
chg_in_log = 1
prev_log = 1000

# #initialize the iteration count
iter_ = 0

#define empty lists to store iteration and cost history into
J_train = []
J_test = []

print("Training the data. Please wait...")

#perform Batch Gradient Decent until convergence criteria is met
while (chg_in_log > log_thresh):
    
    #define the L2 norm regularization term
    L2 = np.linalg.norm(theta)
    #Note to self: We may also want to include the L2 with respect to the the other wieghts. It does not effect the results. Research this!
    for mat in weights:
        L2 += np.linalg.norm(mat)
    
    h_list = [] #train
    h_list_test = [] #test
    
    for i in range(len(weights)):
        if i == 0:
            h = 1/(1 + np.exp(np.dot(-X_stdz_train, weights[i])))
            h_test = 1/(1 + np.exp(np.dot(-X_stdz_test, weights[i])))
            h_list.append(h)
            h_list_test.append(h_test)
            
        else:
            h = 1/(1 + np.exp(np.dot(-h_list[i-1], weights[i]))) 
            h_test = 1/(1 + np.exp(np.dot(-h_list_test[i-1], weights[i])))
            h_list.append(h)
            h_list_test.append(h_test)

    Y_hat_train = 1/(1 + np.exp(np.dot(-h_list[-1], theta)))
    Y_hat_test = 1/(1 + np.exp(np.dot(-h_list_test[-1], theta)))
    
    #compute the error (i.e. the residuals)
    Y_err = (Y_train_onehot - Y_hat_train)
    
    #update the current loss of the cost function for the trianing data
    crnt_log_train = (1/X_stdz_train.shape[0])*(np.sum((Y_train_onehot) * np.log(Y_hat_train + 1e-5) + 
                                                       (1-Y_train_onehot) * np.log(1-Y_hat_train + 1e-5)) - lambda_*L2) 
    
    #update the current loss of the cost function for the trianing data
    crnt_log_test = (1/X_stdz_test.shape[0])*(np.sum((Y_test_onehot) * np.log(Y_hat_test + 1e-5) + 
                                                       (1-Y_test_onehot) * np.log(1-Y_hat_test + 1e-5)) - lambda_*L2)
    
    #append the iterations and costs to the list
    J_train.append((iter_,crnt_log_train))
    J_test.append((iter_,crnt_log_test))
    
    #back propogate and update the beta weights
    for i in range(len(h_list) - 1, -1, -1):
        
        #if only one layer
        if len(h_list) == 1:
            delta_beta = (((Y_err) @ theta.T) * (h_list[i] * (1 - h_list[i]))) 
            weights[i] = weights[i] + eta * (X_stdz_train.T @ delta_beta) - lambda_*weights[i]
            
        else:
            #if greater than one layer
            if i == len(h_list) - 1:
                # delta = dJ/dg(net_0) * dg(net_0)/d(net_0)
                delta_beta = (((Y_err) @ theta.T) * (h_list[i] * (1 - h_list[i]))) 
                weights[i] = weights[i] + eta * (h_list[i-1].T @ delta_beta) - lambda_*weights[i]

            elif i < len(h_list) - 1 and i > 0:
                delta_beta = (delta_beta @ weights[i+1].T) * (h_list[i] * (1-h_list[i])) 
                weights[i] = weights[i] + eta * (h_list[i-1].T @ delta_beta) - lambda_*weights[i] 

            elif i == 0:
                delta_beta =  (delta_beta @ weights[i+1].T) * (h_list[i] * (1-h_list[i])) 
                weights[i] = weights[i] + eta * (X_stdz_train.T @ delta_beta) - lambda_*weights[i]
     
    #compute the gradient with respect to theta
    grad_theta = np.dot(h_list[-1].T, Y_err) - lambda_*theta   

    #perform the update of theta weight
    theta = theta + eta_over_N*grad_theta
    
    #update the change in loss and the previous loss
    chg_in_log = abs((prev_log - crnt_log_train)/prev_log)
    prev_log = crnt_log_train
    
    #increminet the iteration counter
    iter_ += 1

#return the total number of iterations to the screen
print("TOTAL ITERATIONS")
print(iter_)

#### TEST ####

#compute the probabilities of the classification for the testing training sets
h_list_test = []
    
for i in range(len(weights)):
    if i == 0:
        h_test = 1/(1 + np.exp(np.dot(-X_stdz_test, weights[i])))
        h_list_test.append(h_test)     
    else:
        h_test = 1/(1 + np.exp(np.dot(-h_list_test[i-1], weights[i]))) 
        h_list_test.append(h_test)

#compute the Y-prediction values for the test values
Y_hat_test = 1/(1 + np.exp(np.dot(-h_list_test[-1], theta)))

#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_test_pred = np.zeros([Y_hat_test.shape[0], Y_hat_test.shape[1]])  
for i, row in enumerate(Y_hat_test):
    Y_hat_test_pred[i][np.argmax(row)] = 1

#### TRAIN #####

#compute the probabilities of the classification for the testing training sets
h_list_train = []
    
for i in range(len(weights)):
    if i == 0:
        h_train = 1/(1 + np.exp(np.dot(-X_stdz_train, weights[i])))
        h_list_train.append(h_train)     
    else:
        h_train = 1/(1 + np.exp(np.dot(-h_list_train[i-1], weights[i]))) 
        h_list_train.append(h_train)

#compute the Y-prediction values for the train values
Y_hat_train = 1/(1 + np.exp(np.dot(-h_list_train[-1], theta)))

# #creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_train_pred = np.zeros([Y_hat_train.shape[0], Y_hat_train.shape[1]])  
for i, row in enumerate(Y_hat_train):
    Y_hat_train_pred[i][np.argmax(row)] = 1

#keep track of iteration and cost for plots for both train and test
cost_train = []
iteration_train = []
for tup in J_train:
    cost_train.append(tup[1])
    iteration_train.append(tup[0]) #Train
    
cost_test = []
iteration_test = []
for tup in J_test:
    cost_test.append(tup[1])
    iteration_test.append(tup[0]) #Test
    
#confusion matrix train
print("\nTRAIN: CONFUSION MATRIX\n")
conf_train = Y_train_onehot.T @ Y_hat_train_pred
print(conf_train)

#confustion matrix test
print("\nTEST: CONFUSION MATRIX\n")
conf_test = Y_test_onehot.T @ Y_hat_test_pred   
print(conf_test)

#compute the accuracy of the systems
print("\nTRAIN ACCURACY:")
acc_train = np.trace(conf_train)/np.sum(conf_train)
print(acc_train)

print("\nTEST ACCURACY:")
acc_test = np.trace(conf_test)/np.sum(conf_test)
print(acc_test)

#plot the training cost outputs over the iterations for trian
x_train = np.array(iteration_train)
y_train = np.array(cost_train)

fig, axes = plt.subplots()
axes.plot(x_train, y_train, 'b')
axes.set_title("Training Set Cost by Iteration Number")
axes.set_xlabel('Iteration')
axes.set_ylabel('Cost')
_ = plt.savefig("convergence_train_multiANN")

#plot the training cost outputs over the iterations for test
x_test = np.array(iteration_test)
y_test = np.array(cost_test)

fig, axes = plt.subplots()
axes.plot(x_test, y_test, 'b')
axes.set_title("Testing Set Cost by Iteration Number")
axes.set_xlabel('Iteration')
axes.set_ylabel('Cost')
_ = plt.savefig("convergence_test_multiANN")
