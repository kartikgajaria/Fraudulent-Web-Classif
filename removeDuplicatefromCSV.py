import pandas as pd
import csv
parent_dir = "C:\\Users\\kgaja\\Documents\\M.Tech\\Sem 3\\MTP\\myProj\\Archive\\"
""" 
doneFile = open(parent_dir+"\\DoneDomainsPastWHOIS.txt", 'r')
donedata = doneFile.read()
doneWHOISDomains = donedata.split("\n")
while("" in doneWHOISDomains):
    doneWHOISDomains.remove("")
doneFile.close()
i=1
for domain in doneWHOISDomains:
    print(i)
    i=i+1
    path = parent_dir+domain+"\\"
    df1 = pd.read_csv(path+'domainFeatures.csv')
    df1.drop_duplicates(subset=None, inplace=True)
    df1.to_csv(path+'domainFeatures-dpr.csv', index=False)
 """
df1 = pd.read_csv('PastFeaturesCombined.csv')
df1.drop_duplicates(subset=None, inplace=True)
df1.to_csv('PastFeaturesCombined-dpr.csv', index=False)
