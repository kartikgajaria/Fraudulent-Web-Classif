import requests 
import os
import sys
import json
import datetime

parent_dir = "C:\\Users\\kgaja\\Documents\\M.Tech\\Sem 3\\MTP\\myProj\\Archive\\"
ignoreDomain=["psychologytoday.com","linktr.ee","quora.com","typeform.com"]     # some errors associated with these domains

doneFile = open(parent_dir+"\\DoneDomainsURLGen.txt", 'r')
data = doneFile.read()
doneDomains = data.split("\n")
doneFile.close()

domFile = open('finalDomains.txt', 'r')
data = domFile.read()
domains = data.split("\n")
domFile.close()
domains = list(set(domains))

for domain in domains:
    urlList=[]
    if domain in doneDomains:
        continue
    if domain in ignoreDomain:
        continue

    CurrDirectory=domain
    path = parent_dir+CurrDirectory
    print(path)
    f = open(path+"\\"+domain+".json")
    jsonData = json.load(f)
    f.close()
    #print(len(jsonData))
    count=0
    anchorDate=0
    for i in reversed(range(len(jsonData))):
        if(i==0):
            break
        if(count==7):
            break
        if(anchorDate==0):
            anchorDate=jsonData[i][1]
        else:
            if(int(anchorDate)/100000000-int(jsonData[i][1])/100000000>3):
                anchorDate=jsonData[i][1]
            else:
                continue
        if(jsonData[i][4]!='200'):
            continue
        
        #print(jsonData[i])
        count=count+1
        scrapeUrl = "http://web.archive.org/web/"
        timeStamp = jsonData[i][1]
        uUrl = jsonData[i][2]
        scrapeUrl = scrapeUrl+timeStamp+"/"+uUrl
        #print(scrapeUrl)
        urlList.append(scrapeUrl)
    
    with open(path+"\\"+domain+"URLs.txt", 'w') as f:
        for line in urlList:
            f.write(f"{line}\n")
    
    doneDomains.append(domain)

with open(parent_dir+"\\DoneDomainsURLGen.txt", 'w') as f:
    for line in doneDomains:
        f.write(f"{line}\n")
    

print("Done!")