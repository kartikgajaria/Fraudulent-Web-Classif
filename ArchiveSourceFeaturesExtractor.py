from json import dumps, loads
import json
import pandas as pd 
import re
from bs4 import BeautifulSoup
import csv
import glob


fpath = 'C:\\Users\\kgaja\\Documents\\M.Tech\\Sem 3\\MTP\\myProj\\Archive'
folderList = "finalArchiveDomains.txt"

folderFile = fpath+"\\"+folderList
#print(folderFile)
social = ["facebook","instagram","twitter","pinterest"]
fields = ['domain','has_Social','no_Of_External_Links','no_Of_Internal_Links']

with open(folderFile) as f:
    domains = f.read().splitlines()
htmlList = []
#print(len(domains))
for domain in domains:
    domainFolderPath = fpath+"\\"+domain
    
    #print(domainFolderPath)
    htmlPath = domainFolderPath+"\\*.html"
    htmlList.extend(glob.glob(htmlPath))
    #htmlList = [html[-19:-5] for html in htmlList]
    #for html in htmlList:
        #print(html)



""" for htmlFile in htmlList:
        timestamp = htmlFile[-19:-5]
        timestamp = timestamp[:8]
        domainName = htmlFile[57:len(htmlFile)-20]
        print(htmlFile)
        print(timestamp)
 """

rows=[]
i=0
with open(fpath+'\\'+'ArchiveDomainfeatures.csv','w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)


    
    
    for htmlFile in htmlList:
        #print(htmlFile)

        timestamp = htmlFile[-19:-5]
        timestamp = timestamp[:8]
        domainName = htmlFile[57:len(htmlFile)-20]
        #print(htmlFile)
        #print(domainName)
        with open(htmlFile, encoding='utf8', errors='ignore') as line:

        #for line in content:
            extLinks = 0
            intLink = 0
            currDom = domainName
            
            #f.write("%s\n" % line)
            soup = BeautifulSoup(line,features="html.parser",from_encoding="utf8")
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
            newRow = [str(domainName+"-"+timestamp),str(hasSocial),str(extLinks),str(intLink)]
            #csvwriter.writerow(newRow)
            rows.append(newRow)
            i=i+1
        #print(rows)
    csvwriter.writerows(rows)