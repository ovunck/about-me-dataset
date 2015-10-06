# About.me Dataset
 This dataset is generated from user information gathered from [About.me](https://about.me/). About.me is a personal webhosting service, where users can link their social network accounts. My goal for generating this dataset is to gather information about different social network websites for users to analyze their connections.

I used top 100 male and female names from [Social Security Website](http://www.ssa.gov/oact/babynames/decades/century.html) to do a search on About.me with a first name to get a list of About.me usernames. Then for each About.me username, user/view/<username> API call is used to get the information of users. From the user information, social network usernames are extracted for each user and saved as .csv file along other information. Only 4 social media network username is provided for now. (LinkedIn, Facebook, Instagram and Twitter)

## Exploratory Analysis
I performed two exploratory analysis on the about.me dataset:
1. Analyzing the connection between a person's username for different social media networks and whether gender makes a difference for his/her choices. More detailed information can be found [here](http://usernamevsfirstandlastname.herokuapp.com).
2. Predicting social network username based on first and last names. More detailed information can be found [here](http://usernamevsfirstandlastname.herokuapp.com).

## Files
*raw_data: This folder contains the json objects dumped into a single .txt file. Each json object has the
information about About.me user returned as a result of view API call. For more detailed information about APIs, please refer to About.me [API](https://about.me/developer/api/docs/) documentation.

*males.csv, females.csv: This is the csv table generated from json object returned from About.me.

For now the dataset contains information for approximately 2000 users (1100 males and 900 females), due to API call limit of About.me. I'll continue to update this dataset as I gather more user information.

*plot.py: Exploratory analysis code based on Python Data Analysis Library.
