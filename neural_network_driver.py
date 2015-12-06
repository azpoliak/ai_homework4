from neural_network import Neural_Network
import load_data as ld
import pdb

nb = Neural_Network("neural_network",weights = [], num_input=16, num_hidden=1000, num_output=2) #neural_net = Classifier(weights = [], num_input=30, num_hidden=10, num_output=3)

data = ld.load_congress_data(.85)

#data = ld.load_iris(.75)

#data = ld.load_monks(3)

classify =  nb.train(data[0])

#nb.train(iris[0])
#pdb.set_trace()
#nb.test(congress[1])

tot, hit = 0, 0
ones = 0
zeros = 0
twos = 0
for person in data[1]:
  predict = nb.predict(person)
  if predict == person[0]:
  	hit += 1
  tot += 1
  if predict == 1:
  	ones += 1
  elif predict == 0:
  	zeros += 1
  else:
  	twos += 1
  #pdb.set_trace()

print hit, tot, hit / float(tot)
print zeros, ones, twos
