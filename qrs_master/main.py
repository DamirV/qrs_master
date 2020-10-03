import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_loader import QrsDataset
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import drawer


dataset = QrsDataset()
print("success")

for i in range(5):
    y = dataset.__getitem__(0)
    drawer.draw(y)

