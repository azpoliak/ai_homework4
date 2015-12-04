from decision_tree import Decision_Tree
from naive_bayes import Naive_Bayes
import load_data as ld
import pdb

nb = Decision_Tree("decision_tree")

#iris = ld.load_monks(3)
#iris = ld.load_monks(2)
#iris = ld.load_monks(3)
iris = ld.load_iris(.75)
#iris = ld.load_iris_data(.75)

#nb.train(iris[0])
"""tot, hit = 0, 0
for person in iris[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1"""

#classify =  nb.train(iris[0])

nb.train(iris[0])
#nb.train(iris[0])
#nb.train(iris[0])
#nb.train(iris[0])
#pdb.set_trace()
#nb.test(iris[1])


#Accuracy
tot, hit = 0, 0
for person in iris[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1
accuracy = hit/float(tot)
print "Accuracy: ", accuracy

#Recall
id_pos, actual_pos = 0, 0
for person in iris[1]:
  predict = nb.predict(person)
  if predict == iris[1][0][0]:
  	id_pos += 1
  if person[0] == iris[1][0][0]:
        actual_pos += 1
  tot += 1
recall = id_pos/float(actual_pos)
print "Recall: ", recall


#Precision
total, p = 0, 0
for person in iris[1]:
  predict = nb.predict(person)
  if predict == iris[1][0][0]:
    p += 1
  total += 1
precision = id_pos/float(total)
print "Precision: ", precision

"""tot, hit = 0, 0
for person in iris[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1"""

