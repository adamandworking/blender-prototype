import math
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pydot

data = pd.read_csv(r"Dummy_Dataset_1.csv",delimiter=",",header=0)

d=data.values.tolist()

#modify data age>=50
for i in range(len(d)):
    for j in range (len(d[i])):
        if (data.columns[j]=="Age" and d[i][j]!="?"):
#             print (d[i][j])
            if (int(d[i][j])>=50):
                d[i][j]="Age>=50"
            else:
                d[i][j]="Age<50"
            continue
        d[i][j]=data.columns[j] + "=" +str(d[i][j])


            
te = TransactionEncoder()
te_ary = te.fit(d).transform(d)

df = pd.DataFrame(te_ary, columns=te.columns_)

#computing frequent itemsets and association rules
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets

a=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)


for i in range(len(a)):
    if len (a["consequents"][i])>1:
        continue
    if ("Age>=50" in a["antecedents"][i] and "AMD=1" in a["consequents"][i]):
        print (str(a["antecedents"][i]).replace("frozenset","") + " => AMD=1" + " supp: "+ str(  round(a["support"][i],2)  ) + " conf:"+ str(  round(a["confidence"][i],2)  ))


#adding feature names
for i in range(len(d)):
    for j in range (len(d[i])):
        d[i][j]=data.columns[j] + "=" +str(d[i][j])

            
te = TransactionEncoder()
te_ary = te.fit(d).transform(d)

df = pd.DataFrame(te_ary, columns=te.columns_)

#computing frequent itemsets and association rules
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets

a=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)

#visualizing results
print (frequent_itemsets)

print (a[["antecedents","consequents","support","confidence"]])
