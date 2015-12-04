"""
Class for Decision Tree algorithm. Design for decision tree based off of the algorithm outlined in Ch 3 of "Machine
Learning in Action" by Peter Harrington.
"""

import numpy as np
import pdb
import math
import operator

class Decision_Tree:

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
                self.tree = {}
                self.labels = []

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
            #Data is already collected and stored in training_data
            
            #Create labels
            labels = [0] * len(training_data[0])
            i = 0
            for j in range(0, len(labels)):
                labels[j] = i
                i += 1
            self.labels = labels

            #Build tree
            self.tree = self.construct_tree(training_data, self.labels)
            print(self.tree)
            #TODO
            #Report accuracy, precision, and recall using IG and IG ratio
            #Pruning
        
        def construct_tree(self, data, labels):
            """Constucts a tree given the data and attributes."""
            classes = [clas[0] for clas in data]
            #If all labels are the same, return
            if (classes.count(classes[0]) == len(classes)):
                return classes[0]
            #If there is only one attribute left, return
            if (len(data[0]) == 1):
                return self.find_majority(classes)
            #Otherwise, select best feature and recursively build tree
            else:
                best_feature = self.choose_best_split(data)
                best_label = labels[best_feature]
                tree = {best_label:{}}
                del(labels[best_feature])
                feature_values = [clas[best_feature] for clas in data]
                unique_values = set(feature_values)
                for val in unique_values:
                    subset_labels = labels[:]
                    tree[best_label][val] = self.construct_tree(self.split_data_on_attribute(data, best_feature, val), subset_labels)
            
            return tree

        def choose_best_split(self, data):
            """Chooses the best feature tosplit the tree on."""
            numFeatures = len(data[0]) - 1 # Counts how many data features there are
            base_entropy = self.calculate_entropy(data) # Calculates base entropy
            best_info_gain = 0.0 # Keeps track of highest information gain
            best_feature = -1 # Keeps track of feature w/ highest information gain

            #For all entries
            for i in range(0, numFeatures):
                attributeList = [dataset[i] for dataset in data]
                uniqueAttributes = set(attributeList)
                entropy = 0.0
                #Split the data set and note highest gain
                for attribute in uniqueAttributes:
                    subset = self.split_data_on_attribute(data, i, attribute)
                    prob = len(subset)/float(len(data))
                    entropy += prob * self.calculate_entropy(subset)
                info_gain = base_entropy - entropy
                if (info_gain > best_info_gain):
                    best_info_gain = info_gain
                    best_feature = i 
            
            #feature with highest information gain is best feature
            return best_feature

        def find_majority(self, attributeList):
            """Finds the attribute with the majority, given a list."""
            attribute_count = {}
            for attribute in attributeList:
                if attribute not in attribute_count.keys():
                    attribute_count[attribute] = 0.0
                attribute_count[attribute] += 1
            sorted_list = sorted(attribute_count.iteritems(), key=operator.itemgetter(1), reverse=True)
            return sorted_list[0][0]
        
        def split_data_on_attribute(self, data, axis, attribute):
            """Splits the data on a specific attribute and checks if it was worth splitting at that point."""
            split_data = [] # Holds the set of data that splits on the attribute

            #Look through each data point to find attribute
            for array in data:
                if array[axis] == attribute:
                    if type(array).__module__ == np.__name__:
                        as_list = array.tolist()
                    else:
                        as_list = array
                    #Cut out the attribute since it is no longer needed
                    split_array = as_list[:axis]
                    split_array.extend(as_list[axis+1:])
                    split_data.append(split_array)
            return split_data

        def calculate_entropy(self, data):
            """Finds the entropy of a particular attribute in a data set."""
            occurrences = {} # Keeps track of how many times the attribute occurs in the data set
            entropy = 0.0 # Keeps track of the entropy of the data
            data_length = len(data) # Keeps track of how long the data set is

            # Update the count
            for classification in data:
                #If it doesn't exist, add it
                label = classification[0]
                if label not in occurrences.keys():
                    occurrences[label] = 0.0
                occurrences[label] += 1

            #Actually find the entropy
            for occurrence in occurrences:
                prob = occurrences[occurrence]/data_length
                entropy -= prob * math.log(prob/data_length, 2)
            return entropy

        def information_gain(self, data, attribute, attribute2):
            """Determines the information gain that would happen if the particular attribute was chosen."""
            occurrences = {} # Keeps track of how many time the attribute occurs in the data set
            entropy = 0.0 # Keeps track of the entropy of the data

            #Count up occurrences of the attribute
            for record in data:
                #If it does not exist in occurrences, create it
                if not (occurrences.has_key(record[attribute])):
                        occurrences[record[attribute]] = 1.0
                #Otherwise just add 1
                else:
                    occurrences[record[attribute]] += 1

            #Calculate the entropy sum weighted by probability of occurring in data training set
            for occurrence in occurrences.keys():
                prob = occurrences[occurrence] / sum(val_freq.values())
                for record in data:
                    if record[attribute] == val:
                        data_subset = record
                entropy += prob * entropy(data_subset, attribute2)

            #Calculate information gain
            information_gain = entropy(data, attribute2) - entropy
            return information_gain

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
                outcome = self.classify(self.tree, data)
                return outcome

	def test(self, test_data):
		"""
		Data should be nx(m+1) numpy matrix where n is the 
		number of examples and m is the number of features
		(recall that the first element of the vector is the label).

		You should print the accuracy, precision, and recall on the test data.
		"""
                #print(len(test_data))
                results = []
                for i in range(0, len(test_data)):
                    #print("begin")
                    outcome = self.classify(self.tree, test_data[i])
                    results.append(outcome)
                #print(results)
                #print(len(results))

        def find_likely_outcome(self, tree, tally):
            """Finds the most likely outcome for the given tree."""
            #Keep going down each path until a leaf node is reached
            for key in tree.keys():
                #print(key)
                #print(tree[key])
                #traverse if still a tree
                if type(tree[key]).__name__ == 'dict':
                    #print(key)
                    #print(tree.keys())
                    self.find_likely_outcome(tree[key], tally)
                #If not still a tree, we have reached a leaf, so tally leaf count 
                else:
                    #print(tree[key])
                    #print(tally.keys())
                    if tree[key] not in tally.keys():
                        #print("key not in tally")
                        tally[(tree[key])] = 0
                    tally[(tree[key])] += 1
            #Return tally with highest number
            highest_tally = -1
            for item in tally:
                if tally[item] > highest_tally:
                    highest_tally = item
            #print(tally)
            #print(highest_tally)
            return highest_tally

        def classify(self, tree, test_data):
            """Recursively predicts the outcome of the test data."""
            first_label = tree.keys()[0]
            dictionary = tree[first_label]
            classLabel = {}
            #print(first_label)
            #print(test_data)
            #print(dictionary)
            
            for key in dictionary.keys():
                #print (dictionary.keys())
                #print(key)
                #print(test_data[first_label])
                try:
                    #print(test_data[first_label])
                    if test_data[first_label] == key:
                        #print("this was triggered")
                        if type(dictionary[key]).__name__ == 'dict':
                            classLabel = self.classify(dictionary[key], test_data)
                        else:
                            classLabel = dictionary[key]
                #If the label does not exist in the tree's keys
                except :
                    #print(first_label)
                    #print(test_data)
                    #print(test_data[first_label])
                    #print("this was triggered")
                    #check all values below and find most likely one
                    classLabel = self.find_likely_outcome(dictionary, {})
            return classLabel

