import numpy as np
import sys
import load_data as ld
import pdb
from classifier import Classifier

"""
This is the main python method that will be run.
You should determine what sort of command line arguments
you want to use. But in this module you will need to 
1) initialize your classifier and its params 
2) load training/test data 
3) train the algorithm
4) test it and output the desired statistics.
"""
def trainNtest(args):
    classifierType = ["decision_tree", "naive_bayes", "neural_network"]

    
    data = ""
    if len(args) == 4:
        if args[0][3:] == "congress":
            data = ld.load_congress_data(int(args[1][3:]) / 100.0)
        elif args[0][3:] == "monk":
            data = ld.load_monks(int(args[1]))
        elif args[0][3:] == "iris":
            data = ld.load_iris(int(args[1][3:]) / 100.0)
        else:
            print "INVALID DATA NAME"
            return
        method_num = int(args[2][3])
        kwargs = {}
        if method_num == 0 or method_num == 2:
            kwargs[1] = args[2][5]
            kwargs[2] = args[2][7]
        classifier = Classifier(classifierType[int(args[2][3])], one=args[2][5], two=args[2][7])
    else:
        print "ERROR: NEED 4 PARAMETERS"
        return 


    #pdb.set_trace()
    #nb = Naive_Bayes("naive_bayes")

    #classifier = Classifier(classifierType[1])
    #data = ld.load_congress_data(.85)

    #data = ld.load_iris(.70)

    #pdb.set_trace()

    classifier.train(data[0])


    if args[3] == "-test":
        classifier.test(data[1])
    else:
        classifier.test(data[0])

    #classifier.test(data[0])



# just to test the methods 
if __name__=="__main__":

    args = sys.argv[1:]

    trainNtest(args)
    
    #pdb.set_trace()
    #nb.train(iris[0])
    #pdb.set_trace()
    #nb.test(congress[1])
    '''
    tot, hit = 0, 0
    for person in data[1]:
      predict = classifier.predict(person)
      if predict == person[0]:
        hit += 1
      tot += 1

    print hit, tot, hit / float(tot)
    '''

