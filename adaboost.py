# implement a Adaboost classifier from scratch!

class AdaBoostClassifier:
    
    def __init__(self, base_estimator, n_estimaters, learning_rate):
        self.base_estimator = base_estimator
        self.n_estimater = n_estimaters
        self.lr = learning_rate
        
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimater, dtype=np.float)
    
    def fit(self, X, y):
        sample_weight = np.ones(X.shape[0])/X.shape[0]
        for tree in range(self.n_estimater):
            estimator, sample_weight, estimator_weight = self._boost(X, y, sample_weight)
            self.estimators_.append(estimator)
            self.estimator_weight_[tree] = estimator_weight
            
    def _boost(self, X, y, sample_weight):
        estimator = copy.deepcopy(self.base_estimator)
        estimator.fit(X, y, sample_weight=sample_weight)
        pred_y = estimator.predict(X)
        indicator = np.ones(X.shape[0])*[pred_y!=y][0]
        err = np.dot(sample_weight, indicator) / np.sum(sample_weight)
        alpha = np.log((1-err)/err)
        new_sample_weight = sample_weight * np.exp(alpha*indicator)
        return estimator, new_sample_weight, alpha
    
    def predict(self, X):
        predicts = []
        for estimator in self.estimators_:
            pred = estimator.predict(X)
            pred = np.array(pred)
            pred[pred==0] = -1
            predicts.append(pred)
        
        predicts = np.array(predicts)
        
        pr = np.sign(np.dot(self.estimator_weight_, predicts))
        pr[pr==-1] = 0
        return pr.astype(int)