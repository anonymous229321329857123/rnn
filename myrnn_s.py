import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt



class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # This just calls the base class constructor
        super().__init__()
        # Neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically.
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity='tanh', batch_first=True, bias=False)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The RNN also returns its hidden state but we don't use it.
        # While the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default.
        #hh = self.rnn(x)[0]
        hh,h_n = self.rnn(x)
        #x = self.linear(hh[:,x.size()[1]-1])
        x = self.linear(h_n)
        return x

def logistic(r,x):
    return r*x*(1-x)

def shiftt(t,x):
    return np.fmod(x + t + np.random.normal(0,1./9.,x.size).transpose(), 2*np.pi)

# generate n strings of length T from logistic map
def gen_data(n,T):
    output = np.zeros((1,T,n))
    print(output.dtype)
    x = np.random.uniform(0,1,n).transpose()
    output[0] = x
    for t in range(T):
        x = shiftt(0.5,x)
        output[:,t] = x
    return torch.from_numpy(output.transpose())



def train(model_student, model_teacher, data, criterion, optimizer, device):
    # Set the model to training mode. This will turn on layers that would
    # otherwise behave differently during evaluation, such as dropout.
    model_student.train()
    model_teacher.eval()
    # Store the number of sequences that were classified correctly
    num_correct = 0

    # Iterate over every batch of sequences. Note that the length of a data generator
    # is defined as the number of batches required to produce a total of roughly 1000
    # sequences given a batch size.
    #for batch_idx in range(len(data)):

    # Request a batch of sequences and class labels, convert them into tensors
    # of the correct type, and then send them to the appropriate device.
    # data, target = train_data_gen[batch_idx]
    # data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

    # Perform the forward pass of the model
    output = model_student(data)  # Step ①

    # # Pick only the output corresponding to last sequence element (input is pre padded)
    # output = output[:, -1, :] # For many-to-one RNN architecture, we need output from last RNN cell only.

    # Compute the value of the loss for this batch. For loss functions like CrossEntropyLoss,
    # the second argument is actually expected to be a tensor of class indices rather than
    # one-hot encoded class labels. One approach is to take advantage of the one-hot encoding
    # of the target and call argmax along its second dimension to create a tensor of shape
    # (batch_size) containing the index of the class label that was hot for each sequence.
    target = model_teacher(data)

    loss = criterion(output, target)  # Step ②

    # Clear the gradient buffers of the optimized parameters.
    # Otherwise, gradients from the previous batch would be accumulated.
    optimizer.zero_grad()  # Step ③

    loss.backward()  # Step ④

    optimizer.step()  # Step ⑤

    return num_correct, loss.item()

# def test(model, test_data_gen, criterion, device):
#     # Set the model to evaluation mode. This will turn off layers that would
#     # otherwise behave differently during training, such as dropout.
#     model.eval()
#
#     # Store the number of sequences that were classified correctly
#     num_correct = 0
#
#     # A context manager is used to disable gradient calculations during inference
#     # to reduce memory usage, as we typically don't need the gradients at this point.
#     with torch.no_grad():
#         for batch_idx in range(len(test_data_gen)):
#             data, target = test_data_gen[batch_idx]
#             data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)
#
#             output = model(data)
#             # Pick only the output corresponding to last sequence element (input is pre padded)
#             output = output[:, -1, :]
#
#             target = target.argmax(dim=1)
#             loss = criterion(output, target)
#
#             y_pred = output.argmax(dim=1)
#             num_correct += (y_pred == target).sum().item()
#
#     return num_correct, loss.item()


#
# set_default()

def train_and_test(model_student, model_teacher, train_data_gen, criterion, optimizer, max_epochs, i_exp, verbose=True, isPlot=False):
    # Automatically determine the device that PyTorch should use for computation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Move model to the device which will be used for train and test
    model_student.to(device)

    # Track the value of the loss function and model accuracy across epochs
    history_train = {'loss': [], 'acc': []}
    #history_test = {'loss': [], 'acc': []}

    for epoch in range(max_epochs):
        # Run the training loop and calculate the accuracy.
        # Remember that the length of a data generator is the number of batches,
        # so we multiply it by the batch size to recover the total number of sequences.
        num_correct, loss = train(model_student, model_teacher, data, criterion, optimizer, device)
        # accuracy = float(num_correct) / (len(train_data_gen) * train_data_gen.batch_size) * 100
        history_train['loss'].append(np.log(loss))
        # history_train['acc'].append(accuracy)

        # Do the same for the testing loop
        # num_correct, loss = test(model, test_data_gen, criterion, device)
        # accuracy = float(num_correct) / (len(test_data_gen) * test_data_gen.batch_size) * 100
        # history_test['loss'].append(loss)
        # history_test['acc'].append(accuracy)

        if (verbose and epoch%20 == 0) or epoch + 1 == max_epochs:
            print(f'[Experiment {i_exp}, Epoch {epoch + 1}/{max_epochs}]'
                  f" log loss: {history_train['loss'][-1]:.4f}")

    if isPlot:
        # Generate diagnostic plots for the loss and accuracy
        fig, axes = plt.subplots(ncols=1, figsize=(5, 4.5))
        for ax, metric in zip(axes, ['loss']):
            ax.plot(history_train[metric])
            #ax.plot(history_test[metric])
            ax.set_xlabel('epoch', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            #ax.legend(['Train', 'Test'], loc='best')
        plt.show()

    return model_student,history_train['loss']





if __name__ == "__main__":

    batch_size = 4096
    n_batches = 1
    Tseq = 5 #sequence length
    input_size = 1
    output_size = 1
    hidden_size_teacher = 5
    hidden_size_student = 500
    lrate = 0.005
    max_epochs  = 1000
    n_exp = 20

    losscurves = np.zeros(max_epochs*n_exp).reshape(max_epochs,n_exp)


    seed0 = 62
    for i_exp in range(n_exp):
        seed=seed0+i_exp
        np.random.seed(seed)


        model_student = SimpleRNN(input_size, hidden_size_student, output_size)
        model_teacher = SimpleRNN(input_size, hidden_size_teacher, output_size)



        # rescaling model parameter to enter the mean-field regime
        #model_teacher.rnn.weight_hh_l0 = model_teacher.rnn.weight_hh_l0/np.sqrt(hidden_size_teacher)
        for (i,p) in enumerate(model_teacher.rnn.parameters()):
            #print(p)
            if i==1:
                p.data *= 1/np.sqrt(hidden_size_teacher)
        for (i,p) in enumerate(model_teacher.linear.parameters()):
            #print(p)
            p.data *= 1/np.sqrt(hidden_size_teacher)
            #print(p)
        #print("weights whh: "+str(model_teacher.rnn.weight_hh_l0))
        for (i,p) in enumerate(model_student.rnn.parameters()):
            #print(p)
            if i==1:
                p.data *= 10/np.sqrt(hidden_size_student)
        for (i,p) in enumerate(model_student.linear.parameters()):
            #print(p)
            p.data *= 10/np.sqrt(hidden_size_student)
            #print(p)

        criterion = torch.nn.MSELoss()
        optimizer_student = torch.optim.SGD(model_student.parameters(), lr=lrate)


        print("testing data generator:")
        data = gen_data(n=batch_size*n_batches,T=Tseq)
        print("data size (N_sequences, seq_length, input_dim): "+str(data.size()))
        print(data[0])

        model_teacher.double()
        model_student.double()

        output = model_teacher(data)
        #print(output)
        print("data size (N_sequences, seq_length, out_dim): "+str(output.size()))


        _,losscurves[:,i_exp] = train_and_test(model_student, model_teacher, data, criterion, optimizer_student, max_epochs, i_exp)

    np.savetxt("rnn_stoch_hs"+str(hidden_size_student)+"_epochs"+str(max_epochs)+"_lr"+str(lrate)+"_L"+str(Tseq)+"_batchsize"+str(batch_size)+"_seed"+str(seed0)+".txt",losscurves, delimiter=",", fmt='%s')
    fig, ax = plt.subplots()
    ax.plot(losscurves)
    ax.set_xlabel("iterations")
    ax.set_ylabel("MSE")
    plt.show()
