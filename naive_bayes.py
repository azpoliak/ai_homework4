"""
Class for a classification algorithm.
"""
import pdb
import numpy as np
import math

class Naive_Bayes:
 
        
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
        self.classifier_type = classifier_type
        self.params = kwargs
        self.mean = {}
        self.stdv = {}
        self.classify = {}
        """
        The kwargs you inputted just becomes a dictionary, so we can save
        that dictionary to be used in other methods.
        """

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
        #seperate by class
        #pdb.set_trace()
        for row in training_data:
            #print row
            if row[0] not in self.classify:
              self.classify[row[0]] = [row[1:]]
            else:
              #pdb.set_trace()
              self.classify[row[0]].append(row[1:])
        #return classify
        # get mean for each attribute per class
        
        #pdb.set_trace()
        for clas in self.classify.keys():
            for i in range(len(self.classify[clas][0])):
                self.mean[(clas, i)] = np.mean(zip(*self.classify[clas])[i])
                self.stdv[(clas, i)] = np.std(zip(*self.classify[clas])[i])
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
        """
        #realClass = data[0]
        attributes = data[1:]

        #get probablity for each attribute
        prob = {}
        classProb = {}
        for clas in self.classify.keys():
            classProb[clas] = 1
        for i in range(len(attributes)):
            for clas in self.classify.keys():
                mean, stdv = self.mean[(clas, i)], self.stdv[(clas,i)]
                exponent = math.exp(-(math.pow(attributes[i]-mean,2)/(2*math.pow(stdv,2))))
                attributeProb = (1 / math.sqrt(2*math.pi) * stdv) * exponent
                #if i in prob:
                    #prob[i].append(attributeProb)
                    #prob[i][clas] = attributeProb
                #else:
                    #prob[i][clas] = attributeProb
                    #prob[i] = [attributeProb]
                if i not in prob:
                    prob[i] = {}
                prob[i][clas] = attributeProb
        #TODO: change this so that it can allow with cases that have more than 2 classifications
        #pdb.set_trace()
        for attributeProb in prob.values():
            for clas in self.classify.keys():
                classProb[clas] *= attributeProb[clas]
            #classProb[1.0] *= attributeProb[1]
        #classProb[clas] *= attributeProb # classProb[clas] * attributeProb
        #pdb.set_trace()

        #db.set_trace()
        maxClass = self.classify.keys()[0]
        maxProb = classProb[maxClass] 
        for clas in self.classify.keys():
            if classProb[clas] > maxProb:
                maxProb = classProb[clas]
                maxClass = clas
        return maxClass
        



    def test(self, test_data):
        """
        Data should be nx(m+1) numpy matrix where n is the 
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        You should print the accuracy, precision, and recall on the test data.
        """
