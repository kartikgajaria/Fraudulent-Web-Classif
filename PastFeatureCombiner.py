import pandas as pd
import csv
parent_dir = "C:\\Users\\kgaja\\Documents\\M.Tech\\Sem 3\\MTP\\myProj\\Archive\\"
ignDomains = ['grammarchecker.io','carflexi.com']
doneFile = open(parent_dir+"\\DoneDomainsPastWHOIS.txt", 'r')
donedata = doneFile.read()
doneWHOISDomains = donedata.split("\n")
while("" in doneWHOISDomains):
    doneWHOISDomains.remove("")
doneFile.close()
domains = [x for x in doneWHOISDomains if x not in ignDomains]
columns = ['domain','domain_Age','lifeSpan','updatedFor','registrar_Country','host_Country','Same_Registrar_Host_Country','Privacy','TLD','cheap_TLD','domain_Has_Hyphen','domain_Has_Digits','top1m','has_Social','no_Of_External_Links','no_Of_Internal_Links']
i=1
for domain in domains:
    print(domain)
    i=i+1

    DNS_df = pd.read_csv(parent_dir+domain+'\\domainFeatures-dpr.csv')
    SRC_df = pd.read_csv(parent_dir+domain+'\\'+domain+'Features-src.csv')
    """ print(DNS_df)
    print(SRC_df) """
    finalFeatures = []
    for ind in DNS_df.index:
        #domainName = DNS_df['domain'][ind]
        domainAge = DNS_df['domain_Age'][ind]
        lifespan = DNS_df['lifeSpan'][ind]
        updatedFor = DNS_df['updatedFor'][ind]
        registrarC = DNS_df['registrar_Country'][ind]
        hostc = DNS_df['host_Country'][ind]
        samerhc = DNS_df['Same_Registrar_Host_Country'][ind]
        privacy = DNS_df['Privacy'][ind]
        tld = DNS_df['TLD'][ind]
        cheaptld = DNS_df['cheap_TLD'][ind]
        hasHyphen = DNS_df['domain_Has_Hyphen'][ind]
        hasDigits = DNS_df['domain_Has_Digits'][ind]
        top1m = DNS_df['top1m'][ind]
        DNS_date_year=DNS_df['domain'][ind][-6:-2] if DNS_df['domain'][ind][-2:-1] == '-' else DNS_df['domain'][ind][-7:-3]
        DNS_date_month=DNS_df['domain'][ind][-1:] if DNS_df['domain'][ind][-2:-1] == '-' else DNS_df['domain'][ind][-2:]
            
        for idx in SRC_df.index:
            indx = len(SRC_df.index)-(idx+1)
            src_date_year=SRC_df['domain'][indx][-6:-2] if SRC_df['domain'][indx][-2:-1] == '-' else SRC_df['domain'][indx][-7:-3]
            src_date_month=SRC_df['domain'][indx][-1:] if SRC_df['domain'][indx][-2:-1] == '-' else SRC_df['domain'][indx][-2:]
            
            #print(src_date_year + " "+ src_date_month)
            if(SRC_df['domain'][indx]==DNS_df['domain'][ind]):
                hasSocial = SRC_df['has_Social'][indx]
                extLnks = SRC_df['no_Of_External_Links'][indx]
                intLnks = SRC_df['no_Of_Internal_Links'][indx]
                domainName = DNS_df['domain'][ind]+"_"+src_date_year+"-"+src_date_month
                finalFeatures.append([domainName,domainAge,lifespan,updatedFor,registrarC,hostc,samerhc,privacy,tld,cheaptld,hasHyphen,hasDigits,top1m,hasSocial,extLnks,intLnks])
                break
            elif(src_date_year==DNS_date_year):
                hasSocial = SRC_df['has_Social'][indx]
                extLnks = SRC_df['no_Of_External_Links'][indx]
                intLnks = SRC_df['no_Of_Internal_Links'][indx]
                domainName = DNS_df['domain'][ind]+"_"+src_date_year+"-"+src_date_month
                finalFeatures.append([domainName,domainAge,lifespan,updatedFor,registrarC,hostc,samerhc,privacy,tld,cheaptld,hasHyphen,hasDigits,top1m,hasSocial,extLnks,intLnks])
                break
            elif(((int(src_date_year)==int(DNS_date_year)+1) or (int(src_date_year)+1==int(DNS_date_year))) and (abs(int(src_date_month)-int(DNS_date_month))>=3)):
                hasSocial = SRC_df['has_Social'][indx]
                extLnks = SRC_df['no_Of_External_Links'][indx]
                intLnks = SRC_df['no_Of_Internal_Links'][indx]
                domainName = DNS_df['domain'][ind]+"_"+src_date_year+"-"+src_date_month
                finalFeatures.append([domainName,domainAge,lifespan,updatedFor,registrarC,hostc,samerhc,privacy,tld,cheaptld,hasHyphen,hasDigits,top1m,hasSocial,extLnks,intLnks])
                break

    with open(parent_dir+domain+'\\finalCombined.csv', 'w',newline='') as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(columns)
        write.writerows(finalFeatures)
        
