from decision_tree import Decision_Tree
import load_data as ld
import pdb

nb = Decision_Tree("decision_tree")

congress = ld.load_congress_data(.75)

iris = ld.load_iris(.75)

#classify =  nb.train(congress[0])

nb.train(iris[0])
#pdb.set_trace()
#nb.test(congress[1])

tot, hit = 0, 0
for person in iris[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1

#print hit, tot, hit / float(tot)
