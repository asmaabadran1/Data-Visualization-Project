from requests_html import HTMLSession
import csv

session = HTMLSession()
url = session.get("https://cats.com/cat-Breeds")
url.html.render(sleep=2)

products=url.html.xpath("/html/body/div[1]/div/div/article/div/div[4]",first=True)

with open ("about_the_cat.csv",'w',encoding="utf-8",newline='') as file:
    writer=csv.writer(file)
    writer.writerow(["Breed","Description"])
    
    for product in products.absolute_links:
        
        url=session.get(product)
        name=url.html.find("header.entry-header",first=True).text
        #Temperament =url.html.find("div.breed-data-item-value",first=True).text
        Description=url.html.find("div.breed-about-left",first=True).text
        writer.writerow([name,Description])
        
        print(name)
        #print(Temperament)
        print(Description)
    