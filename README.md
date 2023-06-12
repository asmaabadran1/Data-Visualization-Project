# Data-Visualization-Project
😸Purrfect Analytics: A Dashboard for Exploring the World of Cats

this project is a Dashboard about cats' different breeds and their analytics,
This is a Data Visualization project created by Dash and Plotly for Data visualization Course in AI Track - ITI. This project has two main parts : 
1.	insights
2.	a deep learning model that predicts the cat’s breed

This project is divided into 3 main phases
●	phase one → Collecting the Data : 
we used BeautifulSoup, requests_html to scrape this website https://cats.com/cat-breeds for data about the different cat breeds and their characteristics 

we also used https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset 
as an images dataset for our neural network model
and the last part was to preprocess and clean the required dataset for our project



●	Phase Two→Train the model :
preprocess the data for the targeted cats' breed
used pre-trained model inception3 
and used three different optimization methods RMSprop, Adam, and SGD, combined with different learning rates [0.1, 0.01, 0.001]
and trained different combinations between these parameters to get 9 result sets for accuracy and loss, plotted the learning curve for both of them to compare the results of these different models



●	Phase Three→ Designing the dashboard:
we designed a lo-fi sketch for the overall idea in the dashboard in whimsical 
https://whimsical.com/data-visualization-wireframe-8ykwdwMSUKXBK4zVcvjg1


we used a toggle switch to change the mode of our dashboard from insights to predict the breed mode. 
we used dash core components and dash bootstrap components in implementing the design and several callback functions 
to update the dashboard. In the insights mode, we display the top 20 most populous breeds in a bar plot and their respective analytics regarding weight, life expectancy, average body length, and height. We also created an interactive treemap to show the origin of the country for each breed. and their distribution in all continents, with a displayed image of the cat
 For the prediction mode, we used a pre-trained convolutional neural network model to predict the breed of cats image uploaded by users.
 
 # DEMO:
https://drive.google.com/file/d/1FA-iN7NqNvSvXChp83WQ2g8UbwIw-z0A/view 
