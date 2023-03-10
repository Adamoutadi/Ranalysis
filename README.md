# Ranalysis
Project Description

This project analyzes consumer complaints using natural language processing techniques and visualizes the results using various plots and a shiny app. The data used for this analysis is a CSV file containing information about consumer complaints filed with the Consumer Financial Protection Bureau (CFPB).
Libraries

The following libraries were used for this project:
tidyverse: for data manipulation and visualization
tidytext: for text mining and sentiment analysis
dplyr: for data manipulation
wordcloud: for creating word clouds
shiny: for building the shiny app
DT: for rendering the data table in the shiny app
Sentiment Analysis

The sentiment analysis is performed using the NRC and Bing lexicons, which are built-in lexicons in the tidytext package. The NRC lexicon consists of words and their associated emotions, while the Bing lexicon consists of words and their associated positive or negative sentiment.
The analysis first tokenizes the text data and removes stop words. Then, each word is assigned a sentiment based on its presence in the lexicon. The sentiment count is then aggregated by word and sentiment.

<img width="886" alt="Screen Shot 2023-03-09 at 10 42 30 PM" src="https://user-images.githubusercontent.com/75454891/224225357-fc397187-4d79-4590-843f-c7e9bb2df974.png">
<img width="886" alt="Screen Shot 2023-03-09 at 10 46 44 PM" src="https://user-images.githubusercontent.com/75454891/224225933-045ac825-0601-40bf-a3ac-0eeeb328d87b.png">

<img width="886" alt="Screen Shot 2023-03-09 at 10 42 30 PM" src="https://user-images.githubusercontent.com/75454891/224225357-fc397187-4d79-4590-843f-c7e9bb2df974.png">




Graphs and Word Cloud

<img width="886" alt="Screen Shot 2023-03-09 at 10 43 44 PM" src="https://user-images.githubusercontent.com/75454891/224225448-9579ac59-7a05-4cd9-bc48-2654a7793fa3.png">



Two graphs were created to visualize the sentiment count by sentiment and by word count. The first graph shows the top 10 sentiments by count, while the second graph shows the top 100 words by sentiment and word count.
A word cloud was also created using the same sentiment count data. The word cloud displays the top 100 words by sentiment and word count, with positive sentiment words in green and negative sentiment words in red.

<img width="886" alt="Screen Shot 2023-03-09 at 10 45 08 PM" src="https://user-images.githubusercontent.com/75454891/224225687-3ae7e6e8-9cde-4a91-b7ff-f18d229054e1.png">






Shiny App

The shiny app allows the user to choose the X and Y variables for plotting and split the data by a categorical variable. The app includes a scatter plot of the chosen variables and a data table of the filtered data. The user can also download the data table as a CSV file.
The server function uses reactive expressions to update the plot and data table based on the user's input. The renderPlot function renders the scatter plot, while the DT::dataTableOutput function renders the data table.
Overall, this project provides insights into consumer complaints and allows for interactive exploration of the data using a shiny app.
