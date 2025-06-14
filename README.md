### what do i need to import here what will i be working with
- i need a h5py, to extract the data
- i'm not gonna use the scikit-learn class right now because i'll be implementing the model
  by my own.
- that means i need a package/library that can handle matrices.
- i also need a plotting library
- i also need a library that provides mean and variance incase i need to scale the data


```python
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
```

how to extract the data from .h5 file somehow


```python
traindata = h5py.File('./catvnoncat/train_catvnoncat.h5')
train_X = np.array(traindata['train_set_x'])
train_y = np.array(traindata['train_set_y'])
train_X = train_X.reshape((train_X.shape[0], -1));
train_y.resize(train_y.shape[0], 1) 
print("train_x shape: " + str(train_X.shape));
print("train_y shape: " + str(train_y.shape));

testdata = h5py.File("catvnoncat/test_catvnoncat.h5");
test_X = np.array(testdata['test_set_x'])
test_y = np.array(testdata['test_set_y'])
test_X = test_X.reshape(test_X.shape[0], -1);
test_y.resize(test_y.shape[0], 1);
print("test_X shape: " + str(test_X.shape));
print("test_y shape: " + str(test_y.shape));
```

    train_x shape: (209, 12288)
    train_y shape: (209, 1)
    test_X shape: (50, 12288)
    test_y shape: (50, 1)



```python

class LogisticRegression_M:
    NUM_ITERATION = 100
    TOTAL_EXAMPLES = 0
    SIZE = 0
    W = np.zeros((1,1));
    b = np.zeros((1,1));
    w_factor = 0.001;
    b_factor = 0.001;

    def __init__(self, no_iteration, update_factor):
        self.NUM_ITERATION = no_iteration;
        self.w_factor = update_factor;
        self.b_factor = update_factor;

    def _preprocess_input(self, X):
        scaled_X = (X - np.mean(X))/np.std(X)
        return scaled_X

    def __initialize_parameters(self, X, y):
        self.SIZE = X.shape[1:][0];
        self.TOTAL_EXAMPLES = X.shape[0];
        self.W = np.random.rand(self.SIZE, 1)/255;
        #self.b = np.zeros((self.TOTAL_EXAMPLES,1))
        self.b = np.zeros((1,))

    def __sigmoid(self,x):
        return 1/(1 + np.exp(-x) + 1e-8);

    def __calculate_output(self, W, scaled_X, b):
        output = self.__sigmoid(np.dot(scaled_X, W) + b);
        #print(output.shape);
        #output[:10]
        return output;

    def __calculate_loss(self, output, raw_y_array):
        cost = -( raw_y_array*np.log(output) + (1 - raw_y_array)*np.log(1 - output) )
        cost = cost.reshape(-1)
        #print(cost[:10]);
        #print(cost[:10]);
        #print(cost.reshape(-1)[:10]);
        loss = np.sum(cost);
        #print(loss);
        #loss.item();
        return loss;
        

    def __calculate_derivatives(self, scaled_X, output, raw_y_array):
        dw = np.dot(scaled_X.T, output - raw_y_array);
        #print(dw.shape);
        #print(dw[:5]);

        db = np.mean(output - raw_y_array);
        #print(db.shape);
        #print(db[:5]);

        return dw, db;

    def __update_parameters(self, dw, db):
        self.W -= self.w_factor*dw;
        self.b -= self.b_factor*db;

    def fit(self, raw_X_array, raw_y_array):
        losses = []
        scaled_X =  self._preprocess_input(raw_X_array);
        self.__initialize_parameters(raw_X_array, raw_y_array);
        for i in range(self.NUM_ITERATION+1):
            output = self.__calculate_output(self.W, scaled_X, self.b);
            loss = self.__calculate_loss(output, raw_y_array);
            losses.append(i); losses.append(loss.item());
            dw, db = self.__calculate_derivatives(scaled_X, output, raw_y_array);
            self.__update_parameters(dw, db);

            if(i % 10 == 0):
                print(str(i) + ", loss = " + str(loss.item()));

        return self.W, self.b;

    def predict(self, X):
        X = self._preprocess_input(X);
        output = self.__calculate_output(self.W, X, self.b);
        #print(output.shape);
        output = (output > 0.5).astype(int);
        return output;

    
        
```


```python
model_M = LogisticRegression_M(100, 0.001);
model_M.fit(train_X, train_y)
```

    0, loss = 846.1751646186374
    10, loss = 2949.2803693395254
    20, loss = 2406.118804406712
    30, loss = 957.0096983228921
    40, loss = 593.6545204744035
    50, loss = 61.86582457570259
    60, loss = 17.48247993727522
    70, loss = 0.19769686718566792
    80, loss = 0.12601929492699343
    90, loss = 0.1000953446215119
    100, loss = 0.08529906318653134





    (array([[ 0.0867758 ],
            [-0.11150334],
            [-0.06778502],
            ...,
            [-0.02138422],
            [-0.14631137],
            [ 0.144795  ]], shape=(12288, 1)),
     array([-0.00067793]))




```python
output_M = model_M.predict(test_X);
#print(output[:3]);
#print(test_y[:3]);
accuracy = np.sum((output_M == test_y).astype(int))/test_y.shape[0]*100;
print("Accuracy of my model on the test set: " + str(accuracy) + "%");
```

    Accuracy of my model on the test set: 68.0%



```python
from sklearn.linear_model import LinearRegression;

```


```python
model = LinearRegression();
model.fit(train_X, train_y);
output = model.predict(test_X)
output = (output > 0.5).astype(int);
print(output.shape)
accuracy = np.sum((output == test_y).astype(int))/test_y.shape[0]*100;
print("Accuracy of model from sklearn on the test set: " + str(accuracy) + "%");
```

    (50, 1)
    Accuracy of model from sklearn on the test set: 64.0%

