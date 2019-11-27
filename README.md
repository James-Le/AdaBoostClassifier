# AdaBoostClassifier
A simple implementation based on weighted DecisionTreeClassifier both from scratch.

Thanks to [Joachim Valente](https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775) and [xiaoyubai](https://github.com/xiaoyubai/AdaBoost/blob/master/AdaBoostBinary.py), I borrow some code from them, Orz.

If you find the code useful, please give me a star :)

Some coding example below:

```
import numpy as np
from adaboost import AdaBoostClassifier
from base_estimater import Node, DecisionTreeClassifier

myTree = DecisionTreeClassifier(max_depth=10)
myAda = AdaBoostClassifier(myTree, 150, 1)
myAda.fit(X_train, y_train)
y_pred = myAda.predict(X_test)
```
