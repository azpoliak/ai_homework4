from decision_tree import Decision_Tree
from naive_bayes import Naive_Bayes
import load_data as ld
import pdb

nb = Decision_Tree("decision_tree", pruning=False, info_gain_ratio=True)

#congress = ld.load_monks(3)
#congress = ld.load_monks(2)
#congress = ld.load_monks(3)
#congress = ld.load_congress(.75)
congress = ld.load_congress_data(.75)

#nb.train(congress[0])
"""tot, hit = 0, 0
for person in congress[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1"""

#classify =  nb.train(congress[0])

nb.train(congress[0])
#nb.train(congress[0])
#nb.train(congress[0])
#nb.train(congress[0])
#pdb.set_trace()
#nb.test(congress[1])


#Accuracy
tot, hit = 0, 0
for person in congress[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1
accuracy = hit/float(tot)
print "Accuracy: ", accuracy

#Recall
id_pos, actual_pos = 0, 0
for person in congress[1]:
  predict = nb.predict(person)
  if predict == congress[1][0][0]:
  	id_pos += 1
  if person[0] == congress[1][0][0]:
        actual_pos += 1
  tot += 1
recall = id_pos/float(actual_pos)
print "Recall: ", recall


#Precision
total, p = 0, 0
for person in congress[1]:
  predict = nb.predict(person)
  if predict == congress[1][0][0]:
    p += 1
  total += 1
precision = id_pos/float(total)
print "Precision: ", precision

"""tot, hit = 0, 0
for person in congress[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1"""

