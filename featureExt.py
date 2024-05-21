import socket
import requests
import re
import validators
import json
import datetime
from datetime import date
import pandas as pd
import csv
import time
fields = ['domain','domain_Age','lifeSpan','registrar_Country','host_Country','Same_Registrar_Host_Country','Privacy','TLD','cheap_TLD','domain_Has_Hyphen','domain_Has_Digits','top1m']
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

f = open('json\TrustedDomainWhoisData.json')         #<--------------------------------------------change JSON file HERE
data = json.load(f)
df = pd.DataFrame() 
year = 2023

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
with open('TrustedDomainsFeatures13.csv','w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for records in data['whoisRecords'][359:]:
        try:
            if(records['whoisRecord']['dataError']=="MISSING_WHOIS_DATA"):
                continue
        except:
            print()
        
        domainName = records['whoisRecord']['domainName']
        if(domainName is None):
            domainName = records['domainName']
        print(l)
        #domainName = "google.com"
        print("Domain: "+domainName)
        
        # @@@@@@@@@@@@@  DNS Features  @@@@@@@@@@@@@@@@@
        # age feature   
        try:
            creation = records['whoisRecord']['createdDate']
        except:
            try:
                creation = records['whoisRecord']['registryData']['createdDate']
            except:
                creation = "2023-05-22T09:14:53Z"
        try:
            creationDate = datetime.date(int(creation[:4]),int(creation[5:7]),int(creation[8:10]))
        except:
            creationDate = date.today()
        try:
            Domain_Age = records['whoisRecord']['estimatedDomainAge']
            if(Domain_Age is None):
                Domain_Age = calculateAge(creationDate)
            else:
                Domain_Age = Domain_Age/365
        except:
            Domain_Age = calculateAge(creationDate)

        print("Domain age: "+str(Domain_Age))

        #lifeSpan feature
        try:
            expiry = records['whoisRecord']['expiresDate']
        except:
            try:
                expiry = records['whoisRecord']['registryData']['expiresDate']
            except:
                expiry = "2024-06-20T09:14:53Z"
        try:
            expiryDate = datetime.date(int(expiry[:4]),int(expiry[5:7]),int(expiry[8:10]))
        except:
            expiryDate = date.today()
        lifetime = calculateLife(creationDate, expiryDate)
        print("LifeSpan: "+str(lifetime))

        #domain coutnry
        try:
            dCountry = records['whoisRecord']['registrant']['country'].upper()
        except:
            try:
                dCountry = records['whoisRecord']['registryData']['registrant']['country'].upper()
            except:
                dCountry = "NA"
        print("Registered Country: "+dCountry)
        #hosting country
        #hCountry = hostCountry(records['whoisRecord']['domainName']).upper()
        #if(l%10==0):
        #    time.sleep(20)
        hCountry = hostCountry(domainName).upper()
        if(hCountry=="BREAK"):
            break
        l=l+1
        print("Hosted in "+hCountry)

        #same host and domain country
        if(hCountry.__eq__(dCountry)):
            hdsame = True
        else:
            hdsame = False
        print("Same Domain and Host Country? "+str(hdsame))

        
        #domain privacy
        try:
            if(records['whoisRecord']['dataError']):
                privacy = True
        except:
            privacy=False
        print("Privacy enabled? "+str(privacy))
        
        # @@@@@@@@@@@@  URL Features  @@@@@@@@@@@@@@@@@

        #cheap TLD
        #domainName = records['whoisRecord']['domainName']
        
        TLD = re.sub(r'^.*?\.','',domainName)
        #print(domainName)
        print("TLD "+TLD)
        if TLD in cheapTLDset:
            cheapTLD = True
        else:
            cheapTLD = False
        print("Cheap TLD? "+str(cheapTLD))

        # Contains Hyphen
        domainWithoutTLD = domainName.split('.')[0]
        #print("Domain " + domainWithoutTLD)
        if '-' in domainWithoutTLD:
            hasHyphen = True
        else:
            hasHyphen = False
        print("Has Hyphen? "+ str(hasHyphen))

        # Contains Digits
        if bool(re.search(r'\d', domainWithoutTLD))==True:
            hasDigit = True
        else:
            hasDigit = False
        print("Has Digits? "+str(hasDigit))

        # top 1m
        if domainName in set1m:
            top1m = True
        else:
            top1m = False
        print("Top 1M? "+str(top1m))

        
        print("###############################################################")
        new_row=[domainName,str(Domain_Age),str(lifetime),dCountry,hCountry,str(hdsame),str(privacy),TLD,str(cheapTLD),str(hasHyphen),str(hasDigit),str(top1m)]
        #rows.append(new_row)


        csvwriter.writerow(new_row)


#print(df)
f.close()