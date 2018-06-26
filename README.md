# tf_model_lib
Assortment of TensorFlow models

FactorizationMachines:
----------------------
Implementation of original (paper)[https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf] by Steffen Rendel.
```python
from fm.model import FactorizationMachines
from utils import generate_rendle_dummy_dataset

x_data, y_data = generate_rendle_dummy_dataset()
  
fm = FactorizationMachines(l_factors=10)
fm.fit(x_data, y_data)
fm.predict(x_data)
```
