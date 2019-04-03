import pandas as pd
df = pd.read_excel("file1.xlsx")

###
df = pd.read_excel(open("file1.xlsx", "rb"))

###
df = pd.read_excel("file1.xlsx", sheet_name = "Sheet1")
df = pd.read_excel("file1.xlsx", sheet_name = 0)
df = pd.read_excel("file1.xlsx", sheet_name = [0, 1, "Sheet3"])
# import all sheets
df = pd.read_excel("file1.xlsx", sheet_name = None)

###
from sqlalchemy import create_engine
engine = create_engine("sqlite:///:memory:")
df[0].to_sql('data', engine)
df2 = pd.read_sql_table('data', engine)

###
df = pd.read_csv("file1.csv", header = 0)
df = pd.read_csv("file2.csv", header = None, names = ['A', 'B', 'C'])
df = pd.read_csv("file3.csv")