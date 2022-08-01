import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

retail_data = pd.read_csv("online_retail_II.csv")

#STEP 1 PART 1
retail_data = retail_data[retail_data.StockCode.apply(lambda x: x.isnumeric())]
retail_data = retail_data[retail_data.Invoice.apply(lambda x: x.isnumeric())]

#STEP 1 PART 2
retail_data = retail_data.loc[~(retail_data['Price'] < 10)]

#STEP 1 PART 3

allowed_countries = ["United Kingdom", "Italy", "France", "Germany", "Norway", "Finland", "Austria", "Belgium",
"European Community", "Cyprus", "Greece", "Iceland", "Malta", "Netherlands", "Portugal", "Spain", "Sweden",
"Switzerland"]

retail_data = retail_data.loc[(retail_data['Country'].isin(allowed_countries))]

# STEP 1 PART 4
retail_data = retail_data.loc[retail_data['Quantity'] > 0]

#STEP 1 PART 5
retail_data['Description'] = retail_data['Description'].str.strip()

#STEP 2
basket = (retail_data.groupby(['Invoice', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Invoice'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

pd.set_option('display.width', 320)
pd.set_option('display.max_columns',10)

frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)

#STEP 3
rules[rules['confidence'] >= 0.1]
print(rules)