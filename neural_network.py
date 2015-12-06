"""
Class for a classification algorithm.
"""

import numpy as np
import pdb
import math

class Neural_Network:

    def __init__(self, classifier_type, **kwargs):
        """
        Initializer. Classifier_type should be a string which refers
        to the specific algorithm the current classifier is using.
        Use keyword arguments to store parameters
        specific to the algorithm being used. E.g. if you were 
        making a neural net with 30 input nodes, hidden layer with
        10 units, and 3 output nodes your initalization might look
        something like this:

        neural_net = Classifier(weights = [], num_input=30, num_hidden=10, num_output=3)

        Here I have the weight matrices being stored in a list called weights (initially empty).
        """
        np.random.seed(1000)
        self.classifier_type = classifier_type
        self.params = kwargs
        self.classify = {}
        self.weights = self.params['weights']
        self.num_hidden = self.params['num_hidden']
        self.num_input = self.params['num_input']
        self.num_output = self.params['num_output']
        self.layer_input, self.layer_output = [], []
        self.epsilon = .01
        self.reg_lamba = .01

        '''
        Set initial weights ans bias
        wrange = 0.5;
        Whi = wrange/2 * (2 * rand(nhid, ninput) - 1);
        Woh = wrange/2 * (2 * rand(noutput, nhid) - 1);
        bo = wrange/2 * (2 * rand(noutput, 1) - 1);
        bh = wrange/2 * (2 * rand(nhid, 1) - 1);
        '''
        #self.weights = [weights from input to hidden , weights from hidden to output ]
        #self.params['alt_weight'] = 1
        wrange = 1
        if self.params['alt_weight']:
            pdb.set_trace()
            #Glorot & Bengio's weights
            wrange = 1 / math.sqrt(self.num_input)
            
        
        self.weights.append( wrange * (2 * np.random.rand(self.num_hidden, self.num_input) - 1))
        self.weights.append( wrange* (2 * np.random.rand(self.num_output, self.num_hidden) - 1))
        #Delta of first weight, delta of 2nd weight
        self.DW = [0, 0]
        self.train_inputs = np.zeros((369, 16))
        self.train_outputs = np.zeros((369, 1))
        #self.bias = [bias of hidden, bias of output]
        self.bias = [wrange * (2 * np.random.rand(self.num_hidden, 1) - 1), wrange * (2 * np.random.rand(self.num_output, 1) - 1)] 
        #Delta of first bias, delta of 2nd bias
        self.Db = [0, 0]

        #pdb.set_trace()
        
        #pdb.set_trace()
        """
        The kwargs you inputted just becomes a dictionary, so we can save
        that dictionary to be used in other methods.
        """


    def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) 
        
    def dsigmoid(x):
        return x * (1.0 - x)      

    def train(self, training_data):
        """
        Data should be nx(m+1) numpy matrix where n is the 
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        I recommend implementing the specific algorithms in a
        seperate module and then determining which method to call
        based on classifier_type. E.g. if you had a module called
        neural_nets:

        if self.classifier_type == 'neural_net':
            import neural_nets
            neural_nets.train_neural_net(self.params, training_data)

        Note that your training algorithms should be modifying the parameters
        so make sure that your methods are actually modifying self.params

        You should print the accuracy, precision, and recall on the training data.
        """

        lrate = .3
        momentum = 0
        wdecay = 0
        
        for a in range(800):
            #self.train_output = np.zeros((len(training_data), 1))
            terr = 0
            np.random.shuffle(training_data)

            for i in range(len(training_data)):

                #self.train_inputs[i] = training_data[i][1:]
                #self.train_outputs[i] = training_data[i][0]
                curr_input = np.zeros((len(training_data[i][1:]), 1))
                for x in range(len(training_data[i][1:])):
                    curr_input[x] = training_data[i][x+1]
                #sig_hid = np.dot(self.train_inputs[i], self.weights[0])
                #sig_hid = np.add(sig_hid, self.bias[0].T)
                #hidden_layer = self.sigmoid(sig_hid)
                hidden_layer = self.sigmoid(self.weights[0].dot(curr_input) + self.bias[0])

                #sig_out = np.dot(hidden_layer, self.weights[1])
                #sig_out = np.add(sig_out, self.bias[1].T)
                #curr_output = self.sigmoid(sig_out)
                curr_output = self.sigmoid(self.weights[1].dot(hidden_layer) - self.bias[1])

                tt = np.zeros((self.num_output, 1))

                for outputNum in range(self.num_output):
                    #if training_data[i][0] == 1:
                    tt[training_data[i][0]] = 1
                #pdb.set_trace()

                #err = (tt - curr_output).T.dot(tt-curr_output)
                err = -(np.multiply(tt,np.log(curr_output) + np.multiply(1-tt,np.log(1 - curr_output)))).sum()
                #pdb.set_trace()
                terr += err
                #pdb.set_trace()


                '''
                tt = tpat(patno, :)
                Sum of sqaured error
                err = (tt-oo)' * (tt-oo'
                terr = terr + err;
                '''
                
                #err = np.multiply(np.linalg.norm(np.subtract(self.train_outputs, curr_output)),np.subtract(self.train_outputs, curr_output))
                if a > 0:
                    #pdb.set_trace()
                    #backward passing
                    #dtao = np.multiply(np.multiply((tt - curr_output),(curr_output)),(1 - curr_output))
                    #dtah = np.multiply(np.multiply(self.weights[1].T.dot(dtao), hidden_layer),(1 - hidden_layer))
                    dtao = tt - curr_output
                    dtah = np.multiply(np.multiply(self.weights[1].T.dot(dtao), hidden_layer), (1 - hidden_layer))
                    #pdb.set_trace()
                    
                    #weight changes
                    #self.DW[1] = np.subtract(lrate * dtao.dot(hidden_layer.T), self.weights[1].dot(momentum * self.DW[1]))
                    self.DW[1] = (lrate * dtao * hidden_layer.T) - (wdecay * self.weights[1] + momentum * self.DW[1])
                    self.DW[0] = (lrate * dtah * curr_input.T) - (wdecay * self.weights[0] + momentum + self.DW[0])
                    #self.DW[0] = np.subtract(lrate * dtao.dot(curr_output.T), self.weights[1].dot(momentum * self.DW[1]))

                    self.Db[1] = (lrate * dtao * 1) - (wdecay * self.bias[1] + momentum * self.Db[1])
                    self.Db[0] = (lrate * dtah * 1) - (wdecay * self.bias[0] + momentum * self.Db[0])

                    #update weights
                    self.weights[1] += self.DW[1]
                    self.weights[0] += self.DW[0]

                    self.bias[1] += self.Db[1]
                    self.bias[0] += self.Db[0]

                    #pdb.set_trace()


                    '''

                    #dealing with the weight changes
                    blah1 = lrate * dtao * np.linalg.norm(hidden_layer)
                    blah3 = wdecay * self.weights[1] + momentum * self.DW[0]
                    self.DW[0] = np.subtract(lrate * dtao * np.linalg.norm(hidden_layer),wdecay * self.weights[1] + momentum * self.DW[0])
                    blah = lrate * dtah * np.linalg.norm(self.train_inputs[i])
                    blah2 = wdecay * self.weights[0] + momentum * self.DW[1]
                    #pdb.set_trace()
                    #self.DW[1] = np.subtract(lrate * dtah * np.linalg.norm(self.train_inputs[i]), wdecay * self.weights[0] + momentum * self.DW[1])
                    self.Db[0] = np.subtract(lrate * dtao * 1, wdecay * self.bias[0] + momentum * self.Db[0])
                    self.Db[1] = np.subtract(lrate * dtah * 1, wdecay * self.bias[1] + momentum * self.Db[1])

                    # update the weights
                    
                    self.weights[0] += self.DW[1]
                    self.weights[1] += self.DW[0]
                    #pdb.set_trace()
                    #self.bias[0] += self.Db[0]
                    self.bias[1] += self.Db[1]
                    '''
                    





            '''
                % backward pass
                deltao = (tt - oo) .* oo .* (1-oo);
                deltah = (Woh' * deltao) .* hh .* (1-hh);
                
                % weight change
                delta_Woh = lrate * deltao * hh' - wdecay * Woh + momentum * delta_Woh;
                delta_Whi = lrate * deltah * ii' - wdecay * Whi + momentum * delta_Whi;
                delta_bo = lrate * deltao * 1 - wdecay * bo + momentum * delta_bo;
                delta_bh = lrate * deltah * 1 - wdecay * bh + momentum * delta_bh;
                
                % Update weights
                Woh = Woh + delta_Woh;
                Whi = Whi + delta_Whi;
                bo = bo + delta_bo;
                bh = bh + delta_bh;
            '''
            #pdb.set_trace()


            #l1 = self.nonlin(np.dot(self.train_inputs, self.weights[0]))
            #curr_hidden = self.sigmoid(np.add(np.multiply(self.weights[0], curr_input), self.bias[0]))
            #curr_hidden = self.sigmoid(np.add(np.dot(self.weights[0], self.train_inputs), self.bias[0]))
            '''tot1, tot0, correct = 0, 0, 0
            for i in range(len(curr_output)):
                guess = 0
                if curr_output[i][1] > curr_output[i][0]:
                    guess = 1
                    tot1 += 1
                else:
                    tot0 += 1
                if guess == training_data[i][0]:
                    correct += 1'''

            print a

            #pdb.set_trace() 

    def predict(self, data):
        """
        Predict class of a single data vector
        Data should be 1x(m+1) numpy matrix where m is the number of features
        (recall that the first element of the vector is the label).

        I recommend implementing the specific algorithms in a
        seperate module and then determining which method to call
        based on classifier_type.

        This method should return the predicted label.

        ii = ipat(patno, :)';
        hh = sigmoid(Whi * ii + bh);
        oo = sigmoid(Woh * hh + bo);
        """
        classification = data[0]
        data = data[1:]
        curr = np.zeros((len(data), 1))
        for i in range(len(data)):
            curr[i] = data[i]

        hh = self.sigmoid(self.weights[0].dot(curr) + self.bias[0])
        oo = self.sigmoid(self.weights[1].dot(hh) + self.bias[1])
        #pdb.set_trace()
        choice = 0
        maxProb = 0

        blah = oo.argmax()

        for i in range(len(oo)):
            if oo[i] > maxProb:
                maxProb = oo[i]
                choice = i
        #pdb.set_trace()
        return choice


    def test(self, test_data):
        """
        Data should be nx(m+1) numpy matrix where n is the 
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        You should print the accuracy, precision, and recall on the test data.
        """
