import pandas as pd
import numpy as np
import typing
from abc import abstractmethod, ABC


class AbstractAnomalyDetector(ABC):
    @abstractmethod
    def __init__(self, data: pd.Series, interval=None):
        """
        Detect anomalies and get label for each anomaly
        :param data: datetime indexed pd.Series
        :param interval: pair of dates to search for anomalies between
        """
        self.data = data
        if interval:
            self.start, self.end = interval
            if not self.start:
                self.start = 0
            if not self.end:
                self.end = self.data.shape[0]
        else:
            self.start, self.end = 0, self.data.shape[0]
        pass

    @abstractmethod
    def get_labels(self) -> pd.Series:
        pass


class OutlierDetector(AbstractAnomalyDetector):
    """
    Detects anomalies using distribution of data
    Detects:
            Outliers - such points where values are not in 3-sigma range of distribution
                (in other words values are too big or too low than the rest of the data);
    """

    def __init__(self, data: pd.Series, interval=None):
        super().__init__(data, interval)

    def get_labels(self):
        result = self._search_for_anomalies()
        result = result[~np.isnan(result)]
        result = pd.DataFrame(index=result.index, data={'label': result.values})['label']
        if self.start and self.end:
            result = result[self.start:self.end]
        return result

    def _search_for_anomalies(self):
        sigma_min, sigma_max = self.__get_stat()
        return self.data.apply(lambda r: 1 if r > sigma_max else -1 if r < sigma_min else np.nan)

    def __get_stat(self):
        mean_val = self.data.mean()
        std_val = self.data.std()
        sigma_min = mean_val - 3 * std_val
        sigma_max = mean_val + 3 * std_val
        return sigma_min, sigma_max


class AbstractDistributionBasedAnomalyDetector(AbstractAnomalyDetector):

    def __init__(self, data, interval: typing.Tuple[str, str] = None, prev_only: bool = False):
        super().__init__(data, interval)
        self.prev_only = prev_only

    def get_labels(self) -> pd.Series:
        result = self._search_for_anomalies()
        result = pd.DataFrame(index=self.data.index, data={'label': result})['label']
        result = result[~np.isnan(result)]
        return result

    @abstractmethod
    def _search_for_anomalies(self):
        pass


class RareDistributionDetector(AbstractDistributionBasedAnomalyDetector):
    """
    Detects anomalies using distribution of data
    Detects:
            Rare distributions zones -
                such zones where set of values consecutively falls out of range of n*sigma distributions;
    """

    def __init__(self, data, interval: typing.Tuple[str, str] = None, prev_only: bool = False, n=1, window=50):
        super().__init__(data, interval, prev_only)
        self.window = window
        if window > self.end - self.start:
            raise ValueError('Window is bigger than interval')
        if not 0 < n < 4:
            raise ValueError('n must be in (0,4)')
        self.n = n

    def _search_for_anomalies(self):
        length, = self.data.shape
        self.__get_stats(length)
        result = np.array([np.nan] * length)

        for i in range(self.window if self.prev_only else self.start, self.end - self.window + 1):
            is_anomaly = True
            if self.prev_only:
                self.__get_stats(i)
            for w in range(i, i + self.window):
                if is_anomaly:
                    diff = abs(self.data[w] - self.mean)
                    is_anomaly = self.sigma_min < diff < self.sigma_max
            if is_anomaly:
                result[i:i + self.window] = 1
        return result

    def __get_stats(self, i):
        self.mean = self.data[:i].mean()
        self.std = self.data[:i].std()
        self.sigma_min = (self.n - 1) * self.std
        self.sigma_max = self.sigma_min + self.std


class MeanAnomalyDetector(AbstractDistributionBasedAnomalyDetector):
    """
    Detects anomalies using distribution of data
    Detects:
            out-of-mean anomalies - such zones where values are bigger or lower than mean of given data;
    """

    def __init__(self, data, interval: typing.Tuple[str, str] = None, prev_only: bool = False, lower=True, window=50):
        super().__init__(data, interval, prev_only)
        self.window = window
        self.lower = lower
        if window > self.end - self.start:
            raise ValueError('Window is bigger than interval')

    def _search_for_anomalies(self):
        length, = self.data.shape
        self.__get_stats(length)
        result = np.array([np.nan] * length)

        for i in range(self.window if self.prev_only else self.start, self.end - self.window + 1):
            is_anomaly = True
            if self.prev_only:
                self.__get_stats(i)
            for w in range(i, i + self.window):
                if is_anomaly:
                    is_anomaly = self.data[w] < self.mean if self.lower \
                        else self.data[w] > self.mean
            if is_anomaly:
                result[i:i + self.window] = 1
        return result

    def __get_stats(self, i):
        self.mean = self.data[:i].mean()


class DistributionBasedAnomalyDetector(AbstractDistributionBasedAnomalyDetector):
    """
        Detects anomalies using distribution of data
        Detects:
                distributions-change zones -
                    such zones where normal distribution of given data changes;
    """

    def __init__(self, data, interval: typing.Tuple[str, str] = None, prev_only: bool = False, threshold=0.3,
                 window=50):
        super().__init__(data, interval, prev_only)
        self.window = window
        if window > self.end - self.start:
            raise ValueError('Window is bigger than interval')
        self.threshold = threshold

    def _search_for_anomalies(self):
        length, = self.data.shape
        result = np.array([np.nan] * length)
        std_val = self.data.std()
        mean_val = self.data.mean()

        for i in range(self.window if self.prev_only else self.start, self.end - self.window + 1):
            data_slice = self.data[i:i + self.window]
            sl_mean = data_slice.mean()
            sl_std = data_slice.std()
            if self.prev_only:
                prev_slice = self.data[:i]
                mean_val = prev_slice.mean()
                std_val = prev_slice.std()
            is_anomaly = (sl_std > (1 + self.threshold) * std_val or sl_std < (1 - self.threshold) * std_val) \
                         and (sl_mean > (1 + self.threshold) * mean_val or sl_mean < (1 - self.threshold) * mean_val)
            if is_anomaly:
                result[i:i + self.window] = 1
        return result
