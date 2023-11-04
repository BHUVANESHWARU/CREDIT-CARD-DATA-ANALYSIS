import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

#READ THE DATA
a=pd.read_csv('Customer Acqusition.csv')
b=pd.read_csv('Repayment.csv')
c=pd.read_csv('spend.csv')
print(a,b,c)

#head,tail,info,dtypes,shape,isnull
print(a.head(),b.head(),c.head())
print(a.tail(),b.tail(),c.tail())
print(a.info(),b.info(),c.info())
print(a.dtypes,b.dtypes,c.dtypes)
print(a.shape,b.shape,c.shape)
print(a.isnull().sum(),b.isnull().sum(),c.isnull().sum())


#age is less than 18, replace it with mean of age values
df = pd.DataFrame(a)  
df.loc[df['Age']<18,'Age'] = df['Age'].mean()
print(df)

#Merging (Spend and Customer Acquistion) spend amount is more than the limit, replace it with 50% of that customer’s limit
df1 = pd.merge(left = a,
               right = c,
               left_on = 'Customer',
               right_on = 'Customer',
               how = 'inner',
               indicator = True)
Cust_limit = 0.50*df1.Limit
df1.loc[df1['Amount']>df1['Limit'],['Amount','Limit']] = Cust_limit
print(df1)

#Merging (Repayment and Customer Acquistion)the repayment amount is more than the limit, replace the repayment with the limit. 
df2 = pd.merge(left = a,
               right = b,
               left_on = 'Customer',
               right_on = 'Customer',
               how = 'inner',
               indicator = True)
df2.loc[df2['Amount']>df2['Limit'],['Amount','Limit']] = df2.Limit
print(df2)

#unique customers
print(a.Customer.nunique())

#distinct category
print("Product ",df1.Product.nunique())
print("Segment",df1.Segment.nunique())
print("Type",df1.Type.nunique())

#average monthly spend by customers
df1['Month'] = pd.to_datetime(c['Month'])
df1['month'] = df1['Month'].dt.month
g = df1.groupby(["Customer","month"])['Amount'].mean()
print(g)

#AVERAGE MONTHLY REPAYMENT BY CUSTOMERS
df2['Month'] = pd.to_datetime(b['Month'])
df2['month'] = df2['Month'].dt.month
g = df2.groupby(["Customer","month"])['Amount'].mean()
print(g)

#Monthly Rate Of Interest Is 2.9%, What Is The Profit 
profit =(2.9*(df2.Amount - df1.Amount))/100
df2['profit'] = profit
profit_mon = df2.groupby(["month"])['profit'].sum()
print(profit_mon)


print(c.head())
#top 5 product types on which customer is spending
c['Type'].value_counts().head()
c['Type'].value_counts().head(5).plot(kind='bar')
plt.show()

#city is having maximum spend
group_city = df1.groupby(['City'])['Amount'].sum()
print(group_city.nlargest(1))
plt.figure(figsize=(5,10))
group_city.plot(kind="pie",autopct="%1.0f%%",shadow=True,labeldistance=1.0,explode=[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
plt.title("Amount spent on credit card by customers from different cities")
plt.show()

#Age Group Is Spending More Money
group_age = df1.groupby(['Age'])['Amount'].sum()
print(group_age.nlargest(1))

#Top 10 Customers In Terms Of Repayment
print(df2.groupby("Customer")[["Amount"]].sum().sort_values(by="Amount",ascending=False).head(10))

#The City Wise Spend On Each Product On Yearly Basis
df1['Month'] = pd.to_datetime(df1['Month'])
df1['year'] = df1['Month'].dt.year
g = df1.groupby(["City","Product","year"])            
tot_amount = g[["Amount"]].sum().add_prefix("Total_")
print(tot_amount)

#Graph
sns.set()
pd.pivot_table(df1, index =['City','Product'],columns ='year',values ='Amount').plot.bar(figsize=(12,6))
plt.ylabel('Total amount spend')
plt.show()

#Monthly comparison of total spends, city wise
# sns.set()
# pd.pivot_table(df1, index ='Month',columns = 'City',values = 'Amount').plot(kind='bar')
# plt.ylabel("Total amount spend")
df1['Month'] = pd.to_datetime(df1['Month'])
df1['month'] = df1['Month'].dt.month
sns.set()
pd.pivot_table(df1, index ='City',columns ='month',values="Amount").plot(kind='bar',figsize=(8,6))
ax = plt.subplot(111)
ax.legend(loc='upper center', bbox_to_anchor=(1.045,1), shadow=True, ncol=1)
plt.ylabel('Total amount spend')
plt.show()

#Comparison of yearly spend on air tickets
df1['Month'] = pd.to_datetime(df1['Month'])
df1['Year'] = df1['Month'].dt.year
s = df1.loc[(df1.Type == 'AIR TICKET')]
spend = s.groupby(['Year']).aggregate({'Amount':['sum']})
spendgraph=spend.plot.bar(figsize=(8,6))
plt.legend()
plt.ylabel('Total amount spend on Air tickets')
plt.show()


#Comparison of monthly spend for each product 
df1['Month'] = pd.to_datetime(df1['Month'])
df1['month'] = df1['Month'].dt.month
g = df1.groupby(["Product","month"])            
tot_amount = g[["Amount"]].sum().add_prefix("Total_")
tot_amount
#Graph
sns.set()
pd.pivot_table(df1, index ='Product',columns ='month',values ='Amount').plot.bar(figsize=(18,6))
plt.ylabel('Monthly amount spend')
plt.show()


#userdefined with products and (yearly or monthly)with top 10 customers

df2['Month'] = pd.to_datetime(df2['Month'])
df2['Monthly'] = df2['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
df2['Yearly'] = df2['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
def summary_report(product,timeperiod):
    print('Give the product name and timeperiod for which you want the data')
    if product.lower()=='gold' and timeperiod.lower()=='monthly':
        pivot = df2.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Gold',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    elif product.lower()=='gold' and timeperiod.lower()=='yearly':
        pivot = df2.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Gold',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    elif product.lower()=='silver' and timeperiod.lower()=='monthly':
        pivot = df2.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Silver',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    elif product.lower()=='silver' and timeperiod.lower()=='yearly':
        pivot = df2.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Silver',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    if product.lower()=='platinum' and timeperiod.lower()=='monthly':
        pivot = df2.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Platinum',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    elif product.lower()=='platinum' and timeperiod.lower()=='yearly':
        pivot = df2.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Platinum',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    return result
print(summary_report('gold','monthly'))



