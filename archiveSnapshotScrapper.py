import requests 
import os
import sys
import json
import time

parent_dir = "C:\\Users\\kgaja\\Documents\\M.Tech\\Sem 3\\MTP\\myProj\\Archive\\"

domFile = open('finalDomains.txt', 'r')
data = domFile.read()
domains = data.split("\n")
domFile.close()
domains = list(set(domains))

ignFile = open(parent_dir+'\\ignoreDomains.txt', 'r')
igdata = ignFile.read()
ignoreDomain = igdata.split("\n")
ignFile.close()




#ignoreDomain=["psychologytoday.com","linktr.ee","quora.com","typeform.com","nature.com"]

doneFile = open(parent_dir+"\\DoneDomainsScrapping.txt", 'r')
donedata = doneFile.read()
doneDomains = donedata.split("\n")
while("" in doneDomains):
    doneDomains.remove("")
doneFile.close()

for domain in domains:
    temp=domain
    if domain in doneDomains:
        continue
    if domain in ignoreDomain:
        continue
    CurrDirectory=domain
    path = parent_dir+CurrDirectory
    print(path)
    domainUrlsFilePath = path+"\\"+domain+"URLs.txt"
    print(domainUrlsFilePath)
    
    f = open(domainUrlsFilePath, 'r')
    RawUrls = f.read()
    urls = RawUrls.split("\n")
    f.close()
    while("" in urls):
        urls.remove("")

    for url in urls:
        
        print(url)
        timestamp = url[27:41]
        print(timestamp)
        try:
            page = requests.get(url)
        except:
            time.sleep(120)
            try:
                page = requests.get(url)
            except:
                domain=""
                break
        
        print(page.status_code)    
        if(page.status_code!=200):
            time.sleep(120)
            print('not 200')
            page = requests.get(url)
            print(page.status_code)    
        if(page.status_code==200):
            print('is 200')
            #print(page.text)
            with open(path+'\\'+timestamp+'.html', 'wb+') as file:
                file.write(page.content)
        else:
            print('not 200')
            print("!200: "+url)
            domain=""
            break
        
        time.sleep(8)
    if(domain!=""):
        doneDomains.append(domain)
        with open(parent_dir+'\\DoneDomainsScrapping.txt', 'w') as f:
            for line in doneDomains:
                f.write(f"{line}\n")
    else:
        ignoreDomain.append(temp)
        with open(parent_dir+'\\ignoreDomains.txt', 'w') as f:
            for line in ignoreDomain:
                f.write(f"{line}\n")
    print("Progress: "+str(len(doneDomains))+"/"+str(len(domains)))
    time.sleep(40)

print("All Done!")
doneFile.close()