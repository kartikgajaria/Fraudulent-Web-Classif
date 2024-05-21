import requests
import json
import time

parent_dir = "C:\\Users\\kgaja\\Documents\\M.Tech\\Sem 3\\MTP\\myProj\\Archive\\"
domFile = open(parent_dir+'finalArchiveDomains.txt', 'r')
data = domFile.read()
domains = data.split("\n")
domFile.close()
domains = list(set(domains))

evilFile = open('evilDomains.txt', 'r')
evildata = evilFile.read()
evilDomains = evildata.split("\n")
evilFile.close()
evilDomains = list(set(evilDomains))


doneFile = open(parent_dir+'DoneDomainsPastHist.txt', 'r')
donedata = doneFile.read()
doneDomains = donedata.split("\n")
doneFile.close()
doneDomains = list(set(doneDomains))


ignFile = open(parent_dir+'ignoreDomains.txt', 'r')
igdata = ignFile.read()
ignoreDomains = igdata.split("\n")
ignFile.close()
#print(len(domains))
minusIgnDomains = [x for x in domains if x not in ignoreDomains]
minusGoodDomains = [x for x in minusIgnDomains if x in evilDomains]
#print(len(minusIgnDomains))
finalDomains = [x for x in minusGoodDomains if x not in doneDomains]
print(str(len(doneDomains))+' done and '+str(len(finalDomains))+' more left')

#   YOU ARE IN PREVIEW MODE    ////*********///////////////************//////////////

""" URL = "https://whois-history.whoisxmlapi.com/api/v1?apiKey=at_ChGjWA9fbxBkjkSLRMu2vKSdDkXqk"
i=0
for domain in finalDomains[:10]:
    path=parent_dir+domain
    domainName = domain
    mode = 'preview'
    #mode = 'purchase'
    PARAMS = {'domainName':domainName,'mode':mode}

    r = requests.get(url = URL, params=PARAMS)

    json_data = json.loads(r.text)
    #print("***********************************************")
    print("Current Domains: "+domain)
    #print(json_data)

    with open(path+'\\'+domain+'.json', 'w') as outfile:
        outfile.write(json.dumps(json_data,indent=2))

    
    doneDomains.append(domain)
    with open(parent_dir+'DoneDomainsPastHist.txt', 'w') as f:
        for line in doneDomains:
            f.write(f"{line}\n")

    time.sleep(2) """