from dataclasses import dataclass
import os

@dataclass
class StorkImage:
    filename: str
    focus: int
    hour: float
    directory: str
    
    def path(self):
        return os.path.join(self.directory, self.filename)
