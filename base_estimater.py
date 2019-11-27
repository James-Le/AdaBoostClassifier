# implement a decision tree (stump) classifier from scratch!

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    
    def _gini(self, y, sample_weight):
        
        class_weight = [0.0, 0.0]
        
        for i, label in enumerate(y):
            class_weight[label] += sample_weight[i]
        
        return 1.0 - sum((class_weight[c])**2 for c in range(self.n_classes_))
    
    def fit(self, X, y, sample_weight):
        self.N = len(y)
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, sample_weight, 0)
        
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
    
    def _grow_tree(self, X, y, sample_weight, depth=0):
        
        num_samples_per_class = [0, 0]
        for i, label in enumerate(y):
            num_samples_per_class[label] += sample_weight[i]*self.N
        
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y, sample_weight), 
            num_samples=len(y), 
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            )
        
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y, sample_weight)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left, sample_weight_left = X[indices_left], y[indices_left], sample_weight[indices_left]
                X_right, y_right, sample_weight_right = X[~indices_left], y[~indices_left], sample_weight[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, sample_weight_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, sample_weight_right, depth + 1)
        return node
    
    def _best_split(self, X, y, sample_weight):
        
        m = y.size
        if m <= 1:
            return None, None
        
        num_parent = [0, 0]
        for i, label in enumerate(y):
            num_parent[label] += sample_weight[i] * self.N
    
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        
        for idx in range(self.n_features_):
            
            thresholds, classes, weights = zip(*sorted(zip(X[:, idx], y, sample_weight)))
            
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += weights[i-1]*self.N
                num_right[c] -= weights[i-1]*self.N
                
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
               
                gini = (i * gini_left + (m - i) * gini_right) / m
                
                if thresholds[i] == thresholds[i-1]:
                    continue
                
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        
        return best_idx, best_thr