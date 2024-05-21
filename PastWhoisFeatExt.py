import os
import sys
import json
import socket
import requests
import re
import validators
import json
import time
import datetime
from datetime import date
import pandas as pd
import csv


def domainIp(hostname):
    return socket.gethostbyname(hostname)

def hostCountry(domain):
    #time.sleep(0.5)
    ip_address = str(domainIp(domain))
    response = requests.get(f'https://ipapi.co/{ip_address}/json/')
    #print(str(ip_address)+"\n"+str(response.get("country_name")))
    if(response.status_code==430 or response.status_code==429):
        print("API Quota Exceed")
        return "break"
    else:
        response=response.json()
        return response.get("country_name")

def calculateAge(birthDate):
    days_in_year = 365.2425   
    age = int((date.today() - birthDate).days / days_in_year)
    return age
def calculateLife(creationDt, expDt):
    days_in_year = 365.2425   
    age = int((expDt - creationDt).days / days_in_year)
    return age
year = 2024


# Set creation for TOP 1M
df1m = pd.read_csv('csv\\top-1m.csv')
set1m = set()
for i in df1m['domains']:
    set1m.add(i)
    #print(set1m)

# Set Creation for Cheap TLD
cheapTLDset = set()
with open('txt\cheapTLD.txt') as f:
    cheapTLDlist = (f.read().splitlines())
for i in cheapTLDlist:
    cheapTLDset.add(i)
    
rows=[]
l=1
fields = ['domain','domain_Age','lifeSpan','updatedFor','registrar_Country','host_Country','Same_Registrar_Host_Country','Privacy','TLD','cheap_TLD','domain_Has_Hyphen','domain_Has_Digits','top1m']


parent_dir = "C:\\Users\\kgaja\\Documents\\M.Tech\\Sem 3\\MTP\\myProj\\Archive\\"

domFile = open('finalEvilDomains.txt', 'r')
data = domFile.read()
Alldomains = data.split("\n")
domFile.close()
Alldomains = list(set(Alldomains))

doneFile = open(parent_dir+"\\DoneDomainsPastWHOIS.txt", 'r')
donedata = doneFile.read()
doneDomains = donedata.split("\n")
while("" in doneDomains):
    doneDomains.remove("")
doneFile.close()

domains = [x for x in Alldomains if x not in doneDomains]


#print(len(domains))

for domain in domains:
    if(domain in doneDomains):
        continue

    pastJsonFile = open(parent_dir+domain+"\\"+domain+".json", 'r')
    data = json.load(pastJsonFile)
    pastJsonFile.close()
    print("@@@@@@@@@@@@@@@@@@@@@@@")
    print(domain)
    print("@@@@@@@@@@@@@@@@@@@@@@@")

    print(data["recordsCount"])
    with open(parent_dir+domain+'\\domainFeatures.csv','w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        i=1
        for record in data["records"]:
            if(record["domainType"]=="dropped"):
                continue
            if(i==10):
                time.sleep(10)
            domainName = record["domainName"]
            creation = record["createdDateISO8601"] if record["createdDateISO8601"] is not None else record["createdDateRaw"]
            if creation is None:
                continue
            creationDate = datetime.date(int(creation[:4]),int(creation[5:7]),int(creation[8:10]))
            updation = record["updatedDateISO8601"] if record["updatedDateISO8601"] is not None else record["updatedDateRaw"]
            if updation is None:
                continue
            updationDate = datetime.date(int(updation[:4]),int(updation[5:7]),int(updation[8:10]))
            UpdateYear,updateMonth,day = map(int, str(updationDate).split("-"))
            if(UpdateYear<2021):
                break
            """ if(domain=="jstor.org" and UpdateYear==2022 and updateMonth>=12):
                continue """
            CreateYear,month,day = map(int, str(creationDate).split("-"))
            #print("Update Year: "+str(UpdateYear))
            #print("Create Year: "+str(CreateYear))
            Domain_Age = UpdateYear - CreateYear
            print("Domain Age: "+str(Domain_Age))
            expiry = record["expiresDateISO8601"] if record["expiresDateISO8601"] is not None else record["expiresDateRaw"]
            if expiry is None:
                continue
            expiryDate = datetime.date(int(expiry[:4]),int(expiry[5:7]),int(expiry[8:10]))
            ExpiryYear, day, month = map(int, str(expiryDate).split("-"))

            updatedFor = ExpiryYear-UpdateYear
            print("LifeTime left: "+str(updatedFor))
            lifetime = updatedFor+Domain_Age

            dCountry = record["registrantContact"]["country"]
            print("Domain Country: "+str(dCountry))

            nameserversList = record["nameServers"]
            if(len(nameserversList)==0):
                continue
            nameservers = nameserversList[0]
            ns = nameservers.split('|')
            nameServer = ns[0]
            #print(nameServer)
            time.sleep(10)
            try:
                hCountry = (hostCountry(nameServer) if hostCountry(nameServer) is not None else "SUSPENDED").upper()
            except:
                nameServer=ns[1]
                hCountry = (hostCountry(nameServer) if hostCountry(nameServer) is not None else "SUSPENDED").upper()
            if(hCountry=="BREAK"):
                time.sleep(60)
                hCountry = (hostCountry(nameServer) if hostCountry(nameServer) is not None else "SUSPENDED").upper()
                if(hCountry=="BREAK"):
                    print("*********API Quota Problem*************")
                    sys.exit(0)
            print("Host Country: "+str(hCountry))


            if(hCountry==dCountry):
                hdsame = True
            else:
                hdsame = False
            
            print("Same Domain and Host Country? "+str(hdsame))
            
            regName = record["registrantContact"]["name"]
            regStreet = record["registrantContact"]["street"]
            regCity = record["registrantContact"]["city"]
            regPostal = record["registrantContact"]["postalCode"]

            if ((regName == "REDACTED FOR PRIVACY") or (regStreet == "REDACTED FOR PRIVACY") or (regCity == "REDACTED FOR PRIVACY") or (regPostal == "REDACTED FOR PRIVACY")):
                privacy = True
            else:
                privacy = False

            print("Privacy: "+str(privacy))

            TLD = re.sub(r'^.*?\.','',domainName)
            print("TLD: "+TLD)

            if TLD in cheapTLDset:
                cheapTLD = True
            else:
                cheapTLD = False
            print("Cheap TLD: "+str(cheapTLD))

            domainWithoutTLD = domainName.split('.')[0]
            if '-' in domainWithoutTLD:
                hasHyphen = True
            else:
                hasHyphen = False
            print("Has Hyphen: "+ str(hasHyphen))

            # Contains Digits
            if bool(re.search(r'\d', domainWithoutTLD))==True:
                hasDigit = True
            else:
                hasDigit = False
            print("Has Digits: "+str(hasDigit))

            if domainName in set1m:
                top1m = True
            else:
                top1m = False
            print("Top 1M: "+str(top1m))

            new_row=[domainName+"-"+str(UpdateYear)+"-"+str(updateMonth),str(Domain_Age),str(lifetime),str(updatedFor),dCountry,hCountry,str(hdsame),str(privacy),TLD,str(cheapTLD),str(hasHyphen),str(hasDigit),str(top1m)]

            csvwriter.writerow(new_row)
            i=i+1

    doneDomains.append(domain)

    with open(parent_dir+"\\DoneDomainsPastWHOIS.txt", 'w') as f:
        for line in doneDomains:
            f.write(f"{line}\n")

print("Done!")



