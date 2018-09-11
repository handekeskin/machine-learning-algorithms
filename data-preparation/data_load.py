import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('data/data.csv')

print(data)

boy=data[['boy']]

print(boy)

boykilo=data[['boy','kilo']]

print(boykilo)

