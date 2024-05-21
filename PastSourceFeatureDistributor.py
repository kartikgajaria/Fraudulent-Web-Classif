import pandas as pd
import csv
parent_dir = "C:\\Users\\kgaja\\Documents\\M.Tech\\Sem 3\\MTP\\myProj\\Archive\\"

doneFile = open(parent_dir+"\\DoneDomainsPastWHOIS.txt", 'r')
donedata = doneFile.read()
doneWHOISDomains = donedata.split("\n")
while("" in doneWHOISDomains):
    doneWHOISDomains.remove("")
doneFile.close()



df = pd.read_csv(parent_dir+'ArchiveDomainfeatures.csv')

#print(df.head()) 
#df.drop(1,axis=0,inplace=True)

#print(df.index)

for ind in df.index:
    if df['domain'][ind][:-9] not in doneWHOISDomains:
        df.drop(ind,axis=0,inplace=True)
    #df.at[ind,'has_Social']="@"

df.reset_index(drop=True, inplace=True)

for ind in df.index:
    df.at[ind,'domain']=df['domain'][ind][:-4]+"-"+(df['domain'][ind][-4:-2] if int(df['domain'][ind][-4:-2])>9 else df['domain'][ind][-3:-2]) 
#print(df)

columns=['domain','has_Social','no_Of_External_Links','no_Of_Internal_Links']
i=1
for domain in doneWHOISDomains:
    print("Progress: "+str(i))
    i=i+1
    tempList=[]
    for ind in df.index:
        #print(df['domain'][ind][:-(8 if df['domain'][ind][-3:-2] == "-" else 7)])
        if df['domain'][ind][:-(8 if df['domain'][ind][-3:-2] == "-" else 7)] == domain:
            tempList.append([df['domain'][ind],df['has_Social'][ind],df['no_Of_External_Links'][ind],df['no_Of_Internal_Links'][ind]])
    #print(tempList)
    with open(parent_dir+domain+"\\"+domain+"Features-src.csv", 'w',newline='') as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(columns)
        write.writerows(tempList)