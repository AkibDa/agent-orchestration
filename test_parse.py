import pandas as pd

dfs = pd.read_html('kerala_full.html')
df = dfs[3]
print(df.columns.tolist())
print(df.head(2).to_dict('records'))
