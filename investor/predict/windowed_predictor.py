import math
import numpy as np
import investor.predict

from sklearn.metrics import mean_squared_error
from investor.predict.numerics import roi_window, roi

class WindowedPredictor:

    def __init__(self, predictor, sliding, target):
        self.predictor = predictor
        self.sliding   = sliding
        self.target    = target

    def fit(self, x):
        (x, y, _) = self.sliding.slide(x)
        self.predictor.fit(x, y)

    def test(self, x):
        (x, y, bias) = self.sliding.slide(x)
        truth        = bias + y * bias
        prediction   = bias + self.predictor.predict(x) * bias
        rmse         = math.sqrt(mean_squared_error(truth, prediction))
        return truth, prediction, rmse

    def predict(self, win, invest_at):
        x = roi_window(win, invest_at)
        return win[-1] + self.predict(x) * win[-1][self.target]


class KerasPredictor:

    def __init__(self, model, batch, epochs, verbose):
        self.model = model
        self.batch = batch
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, x, y):
        self.model.fit(x, y, batch_size=self.batch, epochs=self.epochs, verbose=self.verbose)

    def predict(self, x):
        prediction = self.model.predict(x)
        return prediction[:, 0]

