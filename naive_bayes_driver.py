from naive_bayes import Naive_Bayes
import load_data as ld
import pdb

nb = Naive_Bayes("naive_bayes")

congress = ld.load_congress_data(.75)

nb.train(congress[0])
nb.test(congress[1])

for person in congress[1]:
  nb.predict(person)
