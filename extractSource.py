from json import dumps, loads
import json
import pandas as pd 
import re
from bs4 import BeautifulSoup
import csv

jsonObj = pd.read_json(path_or_buf="base\dataset.jsonl", lines=True)
contentAll = jsonObj['content'].tolist()
domainData = jsonObj['domain'].tolist()
content=[]
dom=[]
with open("txt\liveDomainsInWayBack.txt") as f:
    dom = (f.read().splitlines())
domains = set()
for i in dom:
    domains.add(i)
#print(domains)
domainsd=[]
a=0
b=0
for x in range(len(domainData)):
    if domainData[x] in domains:
        content.append(contentAll[x])
        domainsd.append(domainData[x])
'''
with open('doma.txt', 'w') as f:
    for line in domainsd:
        f.write(f"{line}\n")       
with open('conta.txt', 'w', encoding="utf-8") as f:
    for line in content:
        f.write(f"{line}\n") 
'''

#print(len(content))
#print(len(domainsd))



#with open('content.txt', 'w') as f:

social = ["facebook","instagram","twitter","pinterest"]
i=0
fields = ['domain','has_Social','no_Of_External_Links','no_Of_Internal_Links']
rows=[]
with open('features.csv','w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

    for line in content:
        extLinks = 0
        intLink = 0
        currDom = domainsd[i].lower()
        
        #f.write("%s\n" % line)
        soup = BeautifulSoup(line,features="html.parser")
        #for tag in soup.findAll('a', href=True):
        #    print(tag['href'])
        hasSocial = False
        for link in soup.findAll('a', attrs={'href': re.compile("^(http|https)://")}):
            #print("############")
            href = link.attrs.get("href")
            if href is not None:
                if (i==1628):
                    print(href)

                #social media
                for media in social:
                    if (media in href) and (currDom not in href):
                        hasSocial = True
                        break
                '''
                if not(hasSocial):
                    hasSocial = re.search(r'/(?:https?:\/\/)?(?:www\.)?(mbasic.facebook|m\.facebook|facebook|fb)\.(com|me)\/(?:(?:\w\.)*#!\/)?(?:pages\/)?(?:[\w\-\.]*\/)*([\w\-\.]*)/ig','https://www.facebook.com/Ancientreasures/')
                    print(hasSocial)
                '''
                if currDom not in href:
                    extLinks=extLinks+1
                else:
                    
                    if any(media in href for media in social):
                        extLinks=extLinks+1
                    else:
                        intLink=intLink+1
        '''
        print("Domain: ",str(domainsd[i-1]))
        print("Has social? ",str(hasSocial))        
        print("External links "+str(extLinks))
        print("Internal links "+str(intLink))
        '''
        newRow = [str(domainsd[i]),str(hasSocial),str(extLinks),str(intLink)]
        #csvwriter.writerow(newRow)
        rows.append(newRow)
        i=i+1
    #print(rows)
    csvwriter.writerows(rows)