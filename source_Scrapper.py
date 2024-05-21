import pandas as pd
import json
import csv
import requests
import re
from bs4 import BeautifulSoup
import csv


with open('txt/liveTrustedDomains.txt') as f:
    liveDomains = f.read().splitlines()



social = ["facebook","instagram","twitter","pinterest"]
i=0
fields = ['domain','has_Social','no_Of_External_Links','no_Of_Internal_Links']
rows=[]
with open('features_Trusted_Live_Domains3.csv','w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for URL in liveDomains[86:]:
        try:
            pageSource = requests.get(f'https://{URL}',timeout=60)
        except:
            try:
                pageSource = requests.get(f'http://{URL}',timeout=60)
            except:
                continue
        line = pageSource.text
    #for line in content:
        extLinks = 0
        intLink = 0
        currDom = URL.lower()
        
        #f.write("%s\n" % line)
        soup = BeautifulSoup(line,features="html.parser")
        #for tag in soup.findAll('a', href=True):
        #    print(tag['href'])
        hasSocial = False
        for link in soup.findAll('a', attrs={'href': re.compile("^(http|https)://")}):
            #print("############")
            href = link.attrs.get("href")
            if href is not None:
                #if (i==1628):
                #    print(href)

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
        print(f'###########################################\nIdx:{i}')
        print("Domain: ",URL.lower())
        print("Has social? ",str(hasSocial))        
        print("External links "+str(extLinks))
        print("Internal links "+str(intLink))
        
        newRow = [URL.lower(),str(hasSocial),str(extLinks),str(intLink)]
        #csvwriter.writerow(newRow)
        rows.append(newRow)
        csvwriter.writerow(newRow)
        i=i+1
    #print(rows)
    #csvwriter.writerows(rows)

    
