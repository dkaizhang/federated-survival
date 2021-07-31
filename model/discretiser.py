import numpy as np

from lifelines import KaplanMeierFitter


class Discretiser():
    """
    Arguments:
    cuts -- number of cuts to make.
    scheme -- either equidistant or km
    """
    def __init__(self, cuts, scheme='equidistant'):
        self._cuts = cuts
        self._scheme = scheme
        self.cuts = None

    def fit(self, durations, events):
        durations = durations.astype(np.float64)
        if self._scheme == 'equidistant':
            self.cuts = np.linspace(0, durations.max(), self._cuts, dtype=np.float64)
        elif self._scheme == 'km':
            kmf = KaplanMeierFitter()
            kmf.fit(durations, events)
            last_duration = durations.max()
            percentiles = np.linspace(kmf.predict(last_duration), kmf.predict(0), self._cuts,dtype=np.float64)
            self.cuts = np.unique([kmf.percentile(p) for p in percentiles][::-1])
        else: 
            ValueError
        return self
    
    def transform(self, durations, events):    
        # -1 so that idx_durations starts from 0
        # +events so that uncensored values never have t=0?
        # clipping it so that max index falls within [0,...,self.cuts - 1]
        idx_durations = np.digitize(durations, self.cuts) - 1 + events
        idx_durations = idx_durations.clip(0, len(self.cuts) - 1)
        return (idx_durations, events)

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        durations, events = self.transform(durations ,events)
        return (durations, events)

    @property
    def dim_out(self):
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts)