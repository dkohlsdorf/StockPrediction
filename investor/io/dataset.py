class DataSet:

    def __init__(self, stoxx):
        '''
        DataSet creation

        :param stoxx: list[(name, dataframe)]
        '''
        self.stoxx = stoxx

    def joined(self):
        (_, start) = self.stoxx[0]
        df = start
        cols = ['Close']
        for k, v in self.stoxx[1:]:
            df = df.join(v, rsuffix='_{}'.format(k))
            cols += ['Close_{}'.format(k)]
        df = df.dropna()
        return df[cols]

