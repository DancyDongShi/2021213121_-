import pandas as pd
from apyori import apriori
dataset = pd.read_csv(r'D:\冲鸭！\上财研究生事务相关\上课相关\研一下\人工智能\作业\Assignment1\Assignment1\data\browsing.txt',header=None)
res = dataset[0].apply(lambda x:x.split(" ")).values
data= []
for list_row in res:
    new_list = list(set([i for i in list_row if i != ""]))
    data.append(new_list)
 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd
te = TransactionEncoder()
#编码
te_ary = te.fit(data).transform(data)   #类似onehot编码，所有的商品都是特征，买了的样本对应1，没买的样本对应0
df = pd.DataFrame(te_ary, columns=te.columns_)
freq=apriori(df,min_support=100/31101, use_colnames=True,max_len=3)
#导入关联规则包
from mlxtend.frequent_patterns import association_rules
#计算关联规则
result = association_rules(freq, metric="confidence", min_threshold=0.4)

    


