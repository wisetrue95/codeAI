import pandas as pd
TEST = "data/Code/test.csv"
SUB = "data/Code/sample_submission.csv"

df = pd.read_csv(TEST)
df1=df[:50000]
df1.to_csv('test_split/test1.csv', index=False)
df2=df[50000:100000]
df2.to_csv('test_split/test2.csv', index=False)
df3=df[100000:140000]
df3.to_csv('test_split/test3.csv', index=False)
df4=df[140000:]
df4.to_csv('test_split/test4.csv', index=False)
print(df)


df = pd.read_csv(SUB)
df1=df[:50000]
df1.to_csv('test_split/sub1.csv', index=False)
df2=df[50000:100000]
df2.to_csv('test_split/sub2.csv', index=False)
df3=df[100000:140000]
df3.to_csv('test_split/sub3.csv', index=False)
df4=df[140000:]
df4.to_csv('test_split/sub4.csv', index=False)
print(df)

