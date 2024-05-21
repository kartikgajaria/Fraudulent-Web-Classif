from concurrent.futures import ThreadPoolExecutor
import requests
from requests.exceptions import ConnectionError
import pandas as pd   
workingdomains=[]
def validate_existence(domain):
    try:
        response = requests.get(f'http://{domain}', timeout=10)
    except ConnectionError:
        print(f'Domain {domain} [---]')
    else:
        print(f'Domain {domain} [+++]')
        workingdomains.append(domain)


#To read from main DB
#jsonObj = pd.read_json(path_or_buf="dataset.jsonl", lines=True)
#dom = jsonObj['domain'].tolist()
#list_domain=[]
#for i in dom[:1500]:
#    list_domain.append(i)
#print(list_domain)

#list_domain = ['google.com', 'facebook.com', 'mediaz.xyz', 'nonexistent_domain.test']

with open('txt/trusted.txt') as f:
    list_of_domains = f.read().splitlines()



with ThreadPoolExecutor() as executor:
    executor.map(validate_existence, list_of_domains)

print("Total Domains: "+str(len(list_of_domains)))
print("Total Live: "+str(len(workingdomains)))


with open('txt\liveTrustedDomains.txt', 'w') as f:
    for line in workingdomains:
        f.write(f"{line}\n")
