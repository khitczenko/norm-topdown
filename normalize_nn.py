import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import csv
import os

# The format of the input txt file is as follows:
# length,duration,f1,f2,f3,contextualfactor_1,...,contextualfactor_n
# The neural net will normalized using all contextual factors.

####################################################################################
################################### Functions ###################################
####################################################################################

# Read in data and prepare for training
def load_file(filename,Ymean,Xmean,Ystd,Xstd,is_training_data):
    arrays = []
    with open(filename, 'r') as f:
        rdr = csv.reader(f)
        header = True
        # Loop through each row in the training input file
        for row in rdr:
            # Skip the header
            if header == True:
                header = False
                continue
            arrays.append([float(i) for i in row])

    Xraw = Variable(torch.FloatTensor(arrays))
    Len = Xraw[:,0]
    Yraw = Xraw[:,1:5]                                  # Get the duration of each training token
    Xraw = Xraw[:,5:]

    if is_training_data:
        Ymean = torch.mean(Yraw,dim=0,keepdim=True)
        Ystd = torch.std(Yraw,dim=0,keepdim=True)
        Xmean = torch.mean(Xraw,dim=0,keepdim=True)
        Xstd = torch.std(Xraw,dim=0,keepdim=True)

    return Len, (Yraw - Ymean) / Ystd, (Xraw - Xmean) / Xstd, Ymean, Xmean, Ystd, Xstd

# Neural net setup
class Net(nn.Module):
    # Create hidden layers and assign them to self
    def __init__(self, input_size, output_size, hiddensizes):
        super(Net, self).__init__()
        # Define hidden layers
        self.h1 = nn.Linear(input_size, hiddensizes[0])
        self.h2 = nn.Linear(hiddensizes[0], hiddensizes[1])
        # self.h3 = nn.Linear(hiddensizes[1], hiddensizes[2])
        # self.h4 = nn.Linear(hiddensizes[2], hiddensizes[3])
        # self.h5 = nn.Linear(hiddensizes[3], hiddensizes[4])
        self.hf = nn.Linear(hiddensizes[-1], output_size)
        self.DEPTH = 3

    # Run forward propagation
    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        # x = F.relu(self.h3(x))
        # x = F.relu(self.h4(x)) 
        # x = F.relu(self.h5(x)) 
        x = self.hf(x)
        # x = F.softmax(x) 
        return x

# Train the network
def train(X,Y,L,X_eval,Y_eval,input_size,output_size,learning_rate,n_epochs,weight_decay,hiddensizes):
    model = Net(input_size = input_size, output_size = output_size,hiddensizes=hiddensizes)
    opt = optim.SGD(params = model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    # Get info for splitting into batches
    N = list(X.size())[0]
    n_batches = N // batch_size

    for epoch in range(n_epochs):
        # Randomly order X, Y, and L (in the same order)
        perm = torch.randperm(N)
        X = X[perm]
        Y = Y[perm]
        L = L[perm]

        total_loss = 0.

        for j in range(n_batches):
            # Get this particular batch
            batchX = X[j:j+batch_size]
            batchY = Y[j:j+batch_size]
            model.zero_grad()

            # Run the model on this batch, calculate loss
            o = model(batchX)
            loss = F.mse_loss(o, batchY)
            total_loss += loss.data[0]

            # Automatically zero the gradient inside the linear layers, update weights
            loss.backward()
            opt.step()

        # Get loss on dev set
        if epoch%max(n_epochs/10,1) == (n_epochs/10 - 1):
            print('loss epoch ' + str(epoch+1) + ':  ' + str(total_loss/n_batches))

        # If last epoch, print overall results
        if epoch == n_epochs - 1:
            o = model(X_eval)
            loss = F.mse_loss(o, Y_eval)
            print 'final train loss: ' + str(total_loss/n_batches)
            print 'final eval loss:  ' + str(loss.data[0])
    return model, total_loss/n_batches, loss.data[0]


####################################################################################
################################### Cross Validating ###################################
####################################################################################

# Choose which data to use
# train_data_file = 'train_pos.txt'
# test_data_file = 'test_pos.txt'
train_data_file = 'train_simppos.txt'
test_data_file = 'test_simppos.txt'
# train_data_file = 'train_nopos.txt'
# test_data_file = 'test_nopos.txt'
# train_data_file = 'train_best.txt'
# test_data_file = 'test_best.txt'

# Parameters to set beforehand
batch_size = 32               ## Batch size
hiddensizes = [10, 10, 10]
learning_rate = 0.001              ## Learning rate
weight_decay = 0.005 # L2 penalty
n_epochs = 100

# Select output directory, end with slash!!
directory_for_data_files = 'nn_simppos/' 

# Notes for ``best'' setting
# A: 
# learning_rate = 0.001
# weight_decay = 0.005

# B: 
# learning_rate = 0.00125
# weight_decay = 0.005

# C: 
# learning_rate = 0.0015
# weight_decay = 0.005

# D: BEST
# learning_rate = 0.00125
# weight_decay = 0.0025

# E: 
# learning_rate = 0.00125
# weight_decay = 0.0075

# F:
# learning_rate = 0.00125
# weight_decay = 0.0015


######################################################################
######################################################################

# # # load standardized train/test data
# L, Y, X, Ymean, Xmean, Ystd, Xstd = load_file(train_data_file,0,0,0,0,is_training_data=True)
# # L_dev, Y_dev, X_dev, Ymean, Xmean, Ystd, Xstd = load_file(test_data_file,Ymean,Xmean,Ystd,Xstd,is_training_data=False)

# # # The set up the file is length, then duration + F1-F3, then remaining contextual factors
# n_train = list(X.size())[0]
# input_size = list(X.size())[1]
# output_size = 1 if len(list(Y.size())) == 1 else list(Y.size())[1]

# # # split train data into 5 parts
# split_size = n_train // 5
# Xs = [X[i*split_size:(i+1)*split_size] for i in range(5)]
# Ys = [Y[i*split_size:(i+1)*split_size] for i in range(5)]
# Ls = [L[i*split_size:(i+1)*split_size] for i in range(5)]

# avg_split_train_loss = 0.
# avg_split_eval_loss = 0.

# for split in range(5):
#     X = torch.cat([Xs[i] for i in range(5) if i != split], dim=0)
#     Y = torch.cat([Ys[i] for i in range(5) if i != split], dim=0)
#     L = torch.cat([Ls[i] for i in range(5) if i != split], dim=0)
#     X_eval = Xs[split]
#     Y_eval = Ys[split]
#     L_eval = Ls[split]

#     # train on 4/5
#     print('training network...')
#     model, train_loss, eval_loss = train(X,Y,L,X_eval,Y_eval,input_size,output_size,learning_rate,n_epochs,weight_decay, hiddensizes)

#     avg_split_train_loss += train_loss/5.
#     avg_split_eval_loss += eval_loss/5.

#     print('calculating residuals...')
#     # get residuals on 4/5 and 1/5
#     residual45 = model(X) - Y
#     residual45 = residual45.data.numpy()
#     residual15 = model(X_eval) - Y_eval
#     residual15 = residual15.data.numpy()

#     print('preparing data...')
#     # make csvs for R
#     data45 = np.concatenate((L.data.numpy().reshape(-1,1),residual45), axis=1)
#     data15 = np.concatenate((L_eval.data.numpy().reshape(-1,1),residual15), axis=1)

#     if not os.path.exists(directory_for_data_files):
#         os.makedirs(directory_for_data_files)

#     print('saving data...')
#     np.savetxt(directory_for_data_files + 'data45_' + str(split+1) + '.csv', data45, delimiter=',')
#     np.savetxt(directory_for_data_files + 'data15_' + str(split+1) + '.csv', data15, delimiter=',')

# # print 'Cross validation train loss of network part: ' + str(avg_split_train_loss)
# # print 'Cross validation eval loss of network part:  ' + str(avg_split_eval_loss)

# # with open('saved_tuning_best.tsv', 'a') as f:
# #     # output_size, learning rate, batch_size, hidden_size, depth
# #     f.write(str(batch_size) + '\t')
# #     f.write(str(hiddensizes) + '\t')
# #     f.write(str(learning_rate) + '\t')
# #     f.write(str(weight_decay) + '\t')
# #     f.write(str(n_epochs) + '\t')
# #     f.write(str(avg_split_train_loss) + '\t')
# #     f.write(str(avg_split_eval_loss) + '\t')
# #     f.write('\n')

#######################################################################
#######################################################################

# train and get residuals for test
L, Y, X, Ymean, Xmean, Ystd, Xstd = load_file(train_data_file,0,0,0,0,is_training_data=True)
L_dev, Y_dev, X_dev, Ymean, Xmean, Ystd, Xstd = load_file(test_data_file,Ymean,Xmean,Ystd,Xstd,is_training_data=False)

# # The set up of the file is length, then duration + F1-F3, then remaining contextual factors
n_train = list(X.size())[0]
input_size = list(X.size())[1]
output_size = 1 if len(list(Y.size())) == 1 else list(Y.size())[1]

print('training network...')
model, train_loss, eval_loss = train(X,Y,L,X_dev,Y_dev,input_size,output_size,learning_rate,n_epochs,weight_decay, hiddensizes)

print('calculating residuals...')
# get residuals
residualtrain = model(X) - Y
residualtrain = residualtrain.data.numpy()
residualtest = model(X_dev) - Y_dev
residualtest = residualtest.data.numpy()

print('preparing data...')
# make csvs to output
datatrain = np.concatenate((L.data.numpy().reshape(-1,1),residualtrain), axis=1)
datatest = np.concatenate((L_dev.data.numpy().reshape(-1,1),residualtest), axis=1)

# If the directory specified for the output doesn't exist, make it
if not os.path.exists(directory_for_data_files):
    os.makedirs(directory_for_data_files)

# Save the csvs for further processing
print('saving data...')
np.savetxt(directory_for_data_files + 'datatrain.csv', datatrain, delimiter=',')
np.savetxt(directory_for_data_files + 'datatest.csv', datatest, delimiter=',')
