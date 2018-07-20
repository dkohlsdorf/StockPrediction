import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models                        import Sequential
from keras.layers                        import Dense

from sklearn.gaussian_process            import GaussianProcessRegressor
from sklearn.gaussian_process.kernels    import RBF
from sklearn.ensemble                    import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors.regression        import KNeighborsRegressor

from investor.io.dataset                 import DataSet
from investor.predict.sliding_window     import SlidingWindow
from investor.predict.windowed_predictor import WindowedPredictor, KerasPredictor

WIN    = 10
LAG    = 5

data = [
    ('euroStoxx50', pd.read_csv('data/stoxx50e.csv', index_col=0, na_values='null').interpolate('linear')),
    ('dax',         pd.read_csv('data/EL4A.F.csv',   index_col=0, na_values='null').interpolate('linear')),
    ('us',          pd.read_csv('data/EL4Z.F.csv',   index_col=0, na_values='null').interpolate('linear')),
    ('xing',        pd.read_csv('data/O1BC.F.csv',   index_col=0, na_values='null').interpolate('linear')),
    ('google',      pd.read_csv('data/GOOGL.csv',    index_col=0, na_values='null').interpolate('linear')),
    ('facebook',    pd.read_csv('data/FB2A.DE.csv',  index_col=0, na_values='null').interpolate('linear')),
    ('amazon',      pd.read_csv('data/AMZN.csv',     index_col=0, na_values='null').interpolate('linear'))
]

data     = DataSet(data)
sequence = data.joined()
dates    = list(sequence.index)
split    = dates[int(len(dates) * 0.8)]

train_sequence = np.array(sequence[split:])
test_sequence  = np.array(sequence[:split])


kernel = RBF(length_scale=2.5)
hidden = [256, 128, 64, 32]
inp    = (len(data.stoxx) * WIN,)
model  = Sequential()
model.add(Dense(hidden[0], activation='relu', input_shape=inp))
for h in hidden[1:]:
    model.add(Dense(h, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile('adam', 'mse')

predictors = [
    ('RF', RandomForestRegressor(n_estimators=250)),
    ('GP', GaussianProcessRegressor(kernel=kernel)),
    ('NN', KNeighborsRegressor(n_neighbors=80)),
    ('NE', KerasPredictor(model, 10, 512, False)),
    ('GB', GradientBoostingRegressor(n_estimators=250))
]                                    

for target in range(0, len(data.stoxx)):
    window = SlidingWindow(WIN, target, LAG)
    plt.figure(num=None, figsize=(10, 5), dpi=200, facecolor='w', edgecolor='k')
    plt.title(
        'Stock: {} from {} to {}'.format(
            data.stoxx[target][0], split, dates[-1]
        )
    )
    i = 0
    for name, base in predictors:
        predictor = WindowedPredictor(base, window, target)
        predictor.fit(train_sequence)
        (truth, predict, rmse) = predictor.test(test_sequence)
        if i == 0:            
            plt.plot(truth, linewidth=1, label='y')
        plt.plot(predict, linewidth=1, label='{} RMSE:   {} Euro'.format(name, int(rmse)))
        i += 1
    plt.xlabel('t')
    plt.ylabel('euro')
    plt.legend()
    plt.savefig('images/{}.png'.format(data.stoxx[target][0]))
    plt.clf()
