import requests
import os
import csv
from bs4 import BeautifulSoup

url = 'https://cats.com/cat-breeds'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')


images = soup.find_all('img')


if not os.path.exists('cat_images'):
    os.makedirs('cat_images')

with open('cat_images.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for image in images:
        image_url = image['src']
        image_name = image_url.split('/')[-1]
        image_path = os.path.join('cat_images', image_name)

        response = requests.get(image_url)
        with open(image_path, mode='wb') as image_file:
            image_file.write(response.content)

        writer.writerow([image_path])