# -*- coding: utf-8 -*-
"""iris_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17kjMJB4zZffY-0p70CATvCVVFO3sfLW3
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
sns.set(style = "white" , color_codes = True)

# df = pd.read_csv('/content/iris.csv')

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/iris.csv')

df.head()

"""# New Section"""

df.plot(kind = "scatter" ,x="SepalLengthCm", y = "SepalWidthCm" )

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df, size=5)

sns.FacetGrid(df, hue="Species", size=5) \
.map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
.add_legend()

sns.boxplot(x="Species", y="PetalLengthCm", data=df)

ax = sns.boxplot(x="Species", y="PetalLengthCm", data=df)

ax = sns.stripplot(x="Species", y="PetalLengthCm", data=df, jitter=True, edgecolor="gray")

sns.violinplot(x="Species", y="PetalLengthCm", data=df, size=6)

sns.FacetGrid(df, hue="Species", size=6) .map(sns.kdeplot, "PetalLengthCm") .add_legend()

sns.pairplot(df.drop("Id", axis=1), hue="Species", size=3)

df.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))

from pandas.plotting import andrews_curves
andrews_curves(df,"Species")

from pandas.plotting import parallel_coordinates
parallel_coordinates(df,"Species")

from pandas.plotting import radviz
radviz(df,"Species")