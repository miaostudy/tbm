import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

train_data_file = "data/merged_tunnel_data.xlsx"
train_data = pd.read_excel(train_data_file)

for col in train_data.columns:
    print(col)

