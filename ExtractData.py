from ucimlrepo import fetch_ucirepo 

#url https://archive.ics.uci.edu/dataset/352/online+retail

# fetch dataset 
online_retail = fetch_ucirepo(id=352) 
  
# data (as pandas dataframes) 
X = online_retail.data.features 
y = online_retail.data.targets 
  
# metadata 
print(online_retail.metadata) 
  
# variable information 
print(online_retail.variables) 
