import os
from stork_v.stork import *

result = Stork().predict_zip_file(
    os.path.join('src', 'data', '78645636.zip'),
    30,
    0)

print(result.lr_eup_anu)
print(result.lr_eup_cxa)