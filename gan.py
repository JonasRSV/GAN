from abc import abstractmethod


class Gan(object):
    @abstractmethod
    def generate(self, samples):
        pass

    @abstractmethod
    def discriminate(self, samples):
        pass

    @abstractmethod
    def update(self, true_samples, false_samples):
        pass
