from dataclasses import dataclass

@dataclass
class StorkData:
    def __init__(self, features, masks): 
          self.features = features
          self.masks = masks