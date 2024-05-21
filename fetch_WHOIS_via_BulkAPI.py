try:
    from urllib.request import Request, urlopen
except ImportError:
    from urllib2 import Request, urlopen
from json import dumps, loads
from time import sleep
import json


from concurrent.futures import ThreadPoolExecutor
import requests
from requests.exceptions import ConnectionError
import pandas as pd   



'''
domains=[]
def validate_existence(domain):
    try:
        response = requests.get(f'http://{domain}', timeout=10)
    except ConnectionError:
        print(f'Domain {domain} [---]')
    else:
        print(f'Domain {domain} [+++]')
        domains.append(domain)


jsonObj = pd.read_json(path_or_buf="dataset.jsonl", lines=True)
dom = jsonObj['domain'].tolist()
list_domain=[]
for i in dom[:15]:
    list_domain.append(i)
print(list_domain)
#list_domain = ['google.com', 'facebook.com', 'mediaz.xyz', 'nonexistent_domain.test']

with ThreadPoolExecutor() as executor:
    executor.map(validate_existence, list_domain)

'''





#jsonObj = pd.read_json(path_or_buf="dataset.jsonl", lines=True)
#dom = jsonObj['domain'].tolist()
#domains = ['ancientreasures.com', 'i-blades.com', 'feelbpay.com']
#for i in dom[:5]:
#    domains.append(i)
#print(domains)

'''
with open('workingDomains.txt', 'w') as f:
    for line in domains:
        f.write("%s\n" % line)
'''


with open('txt\\trusted.txt') as f:
    list_of_domains = f.read().splitlines()

print(len(list_of_domains))


# Your API key
apiKey = 'at_mS4RJQc0j9vvVG4YO25AYFBCUTs9k'

#Input: list of domains to be queried
#domains = ['whoisxmlapi.com', 'threatintelligenceplatform.com']

# Base URL of the API
url = 'https://www.whoisxmlapi.com/BulkWhoisLookup/bulkServices/'
# Encoding of python strings. Should be the default 'utf-8' also for punicode domain names
enc = 'utf-8'
# Interval in seconds between two checks of whether the results are ready
interval = 5

#Making the requests with the domain names, getting the request ID
payload_data = {'domains': list_of_domains, 'apiKey': apiKey, 'outputFormat': 'JSON'}
request = Request(url + 'bulkWhois')
request_id = loads(urlopen(request, dumps(payload_data).encode(enc)).read())['requestId']

#Set the payload for the next phase
payload_data.update({'requestId':request_id, 'searchType':'all', 'maxRecords':1, 'startIndex':1})
del payload_data['domains']

n_domains = len(list_of_domains)
n_remaining = n_domains
request = Request(url + 'getRecords')

while n_remaining > 0:
    print("Waiting %d seconds"%interval)
    sleep(interval)
    response = loads(urlopen(request, dumps(payload_data).encode(enc)).read())
    n_remaining = response['recordsLeft']
    print("%d of %d domains to be processed"%(n_remaining, n_domains))
print("Done, downloading data")
#Get the data
payload_data.update({'maxRecords': n_domains})
result = loads(urlopen(request, dumps(payload_data).encode(enc)).read().decode(enc, 'ignore'))
#print(result)


jsonobj = json.dumps(result, indent=4)


with open("json/TrustedDomainWhoisData.json", "w") as outfile:
    outfile.write(jsonobj)
