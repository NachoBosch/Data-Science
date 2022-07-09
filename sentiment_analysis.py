import os
import pandas as pd
from datetime import datetime
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
nltk.download('vader_lexicon')

with open('C:\Code\Machine_Learning\Data_Science_projects\chat_wp.txt',encoding="utf-8") as f:
    file = f.read()

file = file.split('\n')
file.pop(599)
file.pop(102)
file.pop(103)
file.pop(104)
file.pop(113)
file.pop(159)
file.pop(193)
file.pop(311)
file.pop(436)
file.pop(522)
file.pop(630)
print(type(file),len(file))
print(f"Last row: {file[-1]}")
file.pop(-1)
print(f"Very Last row: {file[-1]}")


#print(len(file))
date = []
day = []
hour = []
names = []
text = []

for f in file:
    date.append(f.split('-')[0])
    # print(f.split('-')[0])
print(f"Len date: {len(date)}")

i=0

for f in file:
    name = f.split('-')[1]
    name = name.split(':')[0]
    names.append(name)
    # print(i,name)
    # print(i,f[:20])
    i+=1
names.pop(0)
names[0]='Pablo Teruya'
names[1]='Pablo Teruya'
names.pop(83)
names[91]='Evelyn Zuloaga'
names[170]='Pablo Teruya'
# [print(i,f) for f,i in enumerate(names)]
print(f"Len names: {len(names)}")

#text
for i,f in enumerate(file):
    # print(i,f.split(':')[-1])
    text.append(f.split(':')[-1])
text.pop(0)
text.pop(0)
text.pop(0)
print(f"Len text: {len(text)}")
print(text[0])
for i in range(1,3):
    text.append(pd.NA)
    names.append(pd.NA)
text.append(pd.NA)
for i in date:
    sp = i.split(' ')
    day.append(sp[0])
    hour.append(sp[1])
print(f"All length: {len(date)}, {len(names)}, {len(text)}, {len(day)}, {len(hour)}")

data = pd.DataFrame({'Hour':hour,'Name':names,'Text':text},index=pd.to_datetime(day))
print(data.head(10))

#Sentimen Analysis
df = data.dropna()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()
df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["Text"]]
df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["Text"]]
df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["Text"]]
print(df.head())

plt.figure()
sns.heatmap(df[["Negative","Positive","Neutral"]].corr(),cmap='coolwarm')
plt.show()












#Eliminar en: [102,104,106,116,163,198,317,433,530,599,640]
# items = [102,104,106,116,163,198,317,433,530,599,640]
# for j in items:
#     print(j)
    
# for n,i in enumerate(date):
#     print(n,i)
# date.pop(102)
# date.pop(104)
# date.pop(106)
# date.pop(116)
# date.pop(163)
# date.pop(198)
# date.pop(317)
# date.pop(433)
# date.pop(530)
# date.pop(599)
# date.pop(604)
# date.pop(104)
# date.pop(103)
# date.pop(112)
# date.pop(158)
# date.pop(192)
# date.pop(104)
# date.pop(310)
# date.pop(434)
# date.pop(521)
# date.pop(589)
# date.pop(514)
# date.pop(110)
# date.pop(154)
# date.pop(186)
# date.pop(302)
# date.pop(424)
# date.pop(575)
# date.pop(612)
# date.pop(628)




