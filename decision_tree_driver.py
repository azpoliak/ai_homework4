from decision_tree import Decision_Tree
from naive_bayes import Naive_Bayes
import load_data as ld
import pdb

#nb = Decision_Tree("decision_tree", pruning=False, info_gain_ratio=True)
nb = Naive_Bayes("naive_bayes")

#monks3 = ld.load_monks(1)
#monks3 = ld.load_monks(2)
#monks3 = ld.load_monks(3)
#monks3 = ld.load_iris(.75)
monks3 = ld.load_congress_data(.75)

#nb.train(monks3[0])
"""tot, hit = 0, 0
for person in monks3[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1"""

#classify =  nb.train(monks3[0])

nb.train(monks3[0])
#nb.train(monks3[0])
#nb.train(monks3[0])
#nb.train(monks3[0])
#pdb.set_trace()
#nb.test(monks3[1])

#Accuracy, Recall, and Precision
relevant_and_retrieved, relevant, retrieved, total, hit = 0, 0, 0, 0, 0
for person in monks3[1]:
  predict = nb.predict(person)
  if predict == monks3[1][0][0] and person[0] == monks3[1][0][0]:
  	relevant_and_retrieved += 1
  if person[0] == monks3[1][0][0]:
        relevant += 1
  if predict == monks3[1][0][0]:
        retrieved += 1
  if predict == person[0]:
        hit += 1
  total += 1
accuracy = hit/float(total)
recall = relevant_and_retrieved/float(relevant)
precision = relevant_and_retrieved/float(retrieved)
print "Accuracy: ", accuracy
print "Precision: ", precision
print "Recall: ", recall

"""tot, hit = 0, 0
for person in monks3[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1"""

