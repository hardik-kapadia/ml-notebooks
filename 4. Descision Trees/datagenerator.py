import numpy as np
import pandas as pd

class Feature:
    
    def __init__(self,name) -> None:
        self.name = name
        
class CategoricalFeature(Feature):
    
    def __init__(self, name, classes: tuple = (0,1)) -> None:
        super().__init__(name)
        self.classes = classes
        
class NumericalFeature(Feature):
    def __init__(self, name, limits) -> None:
        super().__init__(name)
        self.limits = limits

class DataGenerator:
    def __init__(self) -> None:
        self.features = []
        pass
    
    def add_feature(self,feature_name, feature_type, classes: tuple = (0,1), limits: tuple = (0,1)):
        
        if feature_type == 'categorical':
            self.features.append(CategoricalFeature(feature_name,classes))
        elif feature_type =='numerical':
            self.features.append(NumericalFeature(feature_name,limits))
        else:
            raise ValueError("invalid feature type")
        
    def generate_random_data(self, n: int = 5):
        
        data = pd.DataFrame()
        
        for _ in range(n):
            single_column = []