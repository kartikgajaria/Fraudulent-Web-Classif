import requests
import time
import pandas as pd 
with open('txt\workingDomains.txt') as f:
    domains = f.read().splitlines()

#jsonObj = pd.read_json(path_or_buf="base\dataset.jsonl", lines=True)
#dom = jsonObj['domain'].tolist()

#print(type(dom))
print(len(domains))


workingList=[]
count = 0
working = 0
for site in dom:
    x = requests.get('http://archive.org/wayback/available',params={'url':site})
    data=x.json()
    if(data['archived_snapshots']):
        working=working+1
        workingList.append(site)
        #print(data['archived_snapshots']['closest']['url'])
    count=count+1
    if(count%5==0):
        print("Total Count:"+str(count))
        print("Working:"+str(working))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        time.sleep(0.5)

print("Working: "+str(working))
print("TotalL "+str(count))

'''
with open('allDomainsWithWayBack.txt', 'w') as fp:
    for item in workingList:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

'''