from decision_tree import Decision_Tree
import load_data as ld
import pdb

nb = Decision_Tree("decision_tree")

monks1 = ld.load_monks(1)
monks2 = ld.load_monks(2)
monks3 = ld.load_monks(3)
congress = ld.load_congress_data(.75)
iris = ld.load_iris(.75)

nb.train(iris[0])
tot, hit = 0, 0
for person in iris[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1

#classify =  nb.train(congress[0])

#nb.train(monks1[0])
#nb.train(monks2[0])
#nb.train(monks3[0])
#nb.train(iris[0])
#pdb.set_trace()
#nb.test(congress[1])

"""tot, hit = 0, 0
for person in congress[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1"""

"""tot, hit = 0, 0
for person in monks3[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1"""

print hit, tot, hit / float(tot)
