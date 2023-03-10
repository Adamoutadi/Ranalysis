# Libraries
library(tidyverse)
library(tidytext)
library(dplyr)
library(wordcloud)
library(shiny)
library(DT)

# Working directory
setwd("~/Documents/Data332")

# Read in data
df_1 <- read.csv("Consumer_Complaints.csv")
saveRDS(df_1, "Consumer_Complaints.rds")
df <- readRDS("Consumer_Complaints.rds")

# Clean data
df <- df %>%
  select(state = State, company = Company, issue = Issue, company_response = Company.response.to.consumer, product = Product) %>%
  drop_na()

# Tokenize and remove stop words
df_2 <- df %>%
  unnest_tokens(word, company_response) %>%
  anti_join(stop_words)

# Get sentiments and count words by sentiment
nrc <- get_sentiments("nrc")
df_3 <- df_2 %>% 
  inner_join(nrc) %>% 
  group_by(word) %>% 
  count(sentiment, sort = TRUE) 

# Plot sentiment by count
df_3 %>%
  group_by(sentiment) %>%
  slice_max(n, n = 10) %>% 
  ungroup() %>%
  mutate(sentiment= reorder(sentiment, n)) %>%
  ggplot(aes(n, sentiment, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(x = "count", y = "sentiment")

# Plot sentiment by word count
bing <- get_sentiments("bing")
df_3 %>%
  inner_join(bing) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("green", "red"), max.words = 100)

# Set column names for input selections
column_names <- colnames(df)

# Set up UI
ui <- fluidPage( 
  
  titlePanel(title = "Customer Complaints"),
  
  fluidRow(
    column(2,
           selectInput('X', 'Choose X', column_names, column_names[4]),
           selectInput('Y', 'Choose Y', column_names, column_names[2])
    ),
    column(4, plotOutput('plot_01')),
    column(6, DT::dataTableOutput("table_01", width = "100%"))
  )
)

# Set up server
server <- function(input, output){
  
  output$plot_01 <- renderPlot({
    ggplot(df, aes_string(x=input$X, y=input$Y, colour=input$Splitby)) +
      geom_smooth()
  })
}

# Run app
shinyApp(ui=ui, server=server)