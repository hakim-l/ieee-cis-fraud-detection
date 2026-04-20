from abc import ABC, abstractmethod

class BaseColumn(ABC):
    @abstractmethod
    def __init__(self, column_name):
        self.column_name = column_name

class NumericColumn(BaseColumn):
    def __init__(self, column_name, mean, std):
        super().__init__(column_name)
        self.mean = mean
        self.std = std

class CategoricalColumn(BaseColumn):
    def __init__(self, column_name, categories):
        super().__init__(column_name)
        self.categories = categories

class FreeTextColumn(BaseColumn):
    def __init__(self, column_name):
        super().__init__(column_name)