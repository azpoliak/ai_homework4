# ai_homework4
Implementation of ML algorithms for AI Homework4

Adam Poliak, Sayge Schell
Dec 4th, 2015

To run, do the following:

Decision Trees:

python train_and_test.py -d=congress -p=75 -m=0-0-0 -test

Naive Bayes:
python train_and_test.py -d=congress -p=75 -m=1 -test

Neural Networks:
python train_and_test.py -d=congress -p=75 -m=0-0-0 -test


***Notes:***
-d refers to the data set
-p refers to the percent, if -d is monk, then instead of -p=(int), do 1, 2, or 3 to determine with monk data set
-m refers to machine learning method.
    - If m=0, the 2nd number indicates whether or not pruning is set (0 is not set, 1 is set), 3rd number indicates whether or not information gain ration is set (0 for not set, 1 is set)
    - If m=2, the 2nd number indicated whether an alternate weight initilization scheme is used, 3rd number indicated whether momentum is included in computing the error.


**Class Descriptions**
classifier.py - the class the driver program uses to train and test the specific classifier
decision_tree.py - Contains structure for decision tree
naive_bayes.py - Contains structure for naive bayes
neural_network.py - Contains structure for neural network
train_and_test.py - Driver program to use structures

Please note, for precision and recall, '0' was selected for the value for the 'positive' one. All other values were set
to be negatives.

*Decision Trees* -
Key: IGR = Information Gain ratio, IG = Information Gain, NP = No pruning

Congress- 
            IGR-NP          IG - NP       
Accuracy:   0.935779815614  0.926605504587
Precision:  0.931818181818  0.911111111111
Recall:     0.911111111111  0.911111111111

Monk 1 -

Accuracy:   0.916666666667  0.796296296296
Precision:  1.0             1.0
Recall:     0.888888888889  0.703703703704

Monk2 -

Accuracy:  0.530092592593   0.511574074074
Precision: 0.741228070175   0.757281553398
Recall:    0.58275862069    0.537931034483

Monk3 -

Accuracy:   0.91666666667   0.925925925926
Precision:  0.941176470588  0.96
Recall:     0.941176470588  0.941176470588

Iris-

Accuracy:   0.526315789474  0.710526315789
Precision:  0.75            1.0
Recall:     0.75            0.916666666667

In general, using the information gain ratio will raise the accuracy, recall, and precision of the predictions for the
test data. That is, generally, it is better to use the information gain ratio to determine the split in the tree rather
than just the pure information gain. However, in the case of Iris and Monk3, accuracy, precision, and recall were all
lowered when using the information gain ratio, suggesting that using this is not necessarily better in all cases.

We did not end up fully implementing pruning, but if we were to, the accuracy would most likely remain the same, the
precision would go up, and the recall would go down.

*Naive Bayes*-

            Congress            Monk1           Monk2           Monk3           Iris
Accuracy:   .889908256881       .576388888889   .643259259259   .731481481481   .947368421053
Precision:  .971428571429       .648648648649   .674603174603   .6375           1.0
Recall:     .755555555556       .333333333333   .879310344828   1.0             1.0

*Neural Network* - 
For the neural network we seed our random value by 1000. After trial and error, this gave us the best results. We
also use 800 epochs. We found that any thing larger than 800 doesnt affect our neural network too much

Key: DW = default weight, AW = alternate weight, M = with momentum, NM = without momentum
            DW-NM           AW-NM           DW-M            AW-M
Congress-    
Accuracy:  0.880733944954   0.779816513761  0.412844036697  0.48623853211
Precision  0.921052631579   0.652173913043  0.412844036697  0.445544554455
Recall:    0.777777777778   1.0             1.0             1.0


Monk1 -
Accuracy:  0.814814814815   0.5             0.75462962963   0.511574074074
Precision  1.0              0.5             0.991071428571  0.505854800937
Recall:    0.62962962963    1.0             0.513888888889  1.0


Monk2 -     
Accuracy:   0.671296296296  breaks          0.671296296296  breaks with AW
Precision:  0.671296296296  with            0.671296296296
Recall:     1.0             AW              1.0

Monk3 -
Accuracy:  0.895833333333   0.569444444444  0.75             breaks with AW
Precision  0.841201716738   0.875           0.658940397351   and M
Recall:    0.960784313725   0.102941176471  0.975490196078

Iris-
Accuracy:  0.894736842105   0.921052631579  1.0              0.921052631579
Precision  1.0              1.0             1.0              1.0
Recall:    1.0              1.0             1.0              1.0


*Best Algorithms for each data set*
Congress: Decision Tree (w/ information gain ratio)
Monk1: Neural Network (w/ default weight, no momentum)
Monk2: Neural Network (w/ default weight, no momentum)
Monk3: Decision Tree (w/ information gain)
Iris: Neural Network (w/ default weight, with momentum)

We decided on the best for each data set by determining the method that provided the highest average of accuracy, precision, and recall.
If we look at the best methods for each, they make some logical sense. For example, it makes sense that Congress would be best with a
decision tree. It is likely that one of the criteria is most important and would sway the vote. It is also likely that there is a second
most important criteria that can sway the vote. That is, people naturally tend to categorize political issues by importance, and will vote
based on their personal ranking of importance. This is exactly what the tree is doing. Additionally, it makes sense that the Iris data would
best work with a neural network. A neural network is supposed to mimic a brain and create connections between certain input and certain output.
There is no single 'most important' factor that makes an Iris an Iris. Rather, it is the entire set of factors that come together that can
allow the Iris to be recognized as the specific Iris. This neural network works like a brain to recognize the type of Iris.


Work Breakdown: 
Naive Bayes: Adam
Decision Trees: Sayge
Neural Networks + Decision Tree Pruning + Data: Adam + Sayge

Bugs: We did not fully implement pruning. We were able to count all occurrences for each path using the data, calculate
the chi-squared number, and perform the significance test to decide whether or not to prune. However, we had problems
traversing our data structure because we did not use a node class (just a dictionary). We also noticed this gave us some
weird leafs to collapse. (In one of our cases, it said the head node should be collapsed into one of it's children,
which did not seem correct). If we had implemented the structure with nodes, then pruning would have been much easier.
