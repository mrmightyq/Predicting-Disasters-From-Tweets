## Importing packages

# This R environment comes with all of CRAN and many other helpful packages preinstalled.
# You can see which packages are installed by checking out the kaggle/rstats docker image: 
# https://github.com/kaggle/docker-rstats

library(tidyverse) # metapackage with lots of helpful functions

## Running code

# In a notebook, you can run a single code cell by clicking in the cell and then hitting 
# the blue arrow to the left, or by clicking in the cell and pressing Shift+Enter. In a script, 
# you can run code by highlighting the code you want to run and then clicking the blue arrow
# at the bottom of this window.

## Reading in files

# You can access files from datasets you've added to this kernel in the "../input/" directory.
# You can see the files added to this kernel by running the code below. 

#Load library and train data
library(readr)
library(caret)

library(dplyr)

#Text mining packages
library(tm)
library(SnowballC)

#loading the data
#combine train and test data to make preprocessing easier
disaster_train <- read_csv("C:/Users/KnudseQ/Desktop/disaster train.csv")
disaster_df <- disaster_train
test_df <- read_csv("C:/Users/KnudseQ/Desktop/disaster test.csv")

summary(disaster_train)


library(sentimentr)
sent <- sentiment(disaster_train$text)
sent1 <- as.data.frame(sent)
sent1
sent2 <- aggregate(sent1 [, 4], list(sent1$value.value.element_id), mean)
colnames(sent2)<-c("Id", "Sentiment_Score")
sent_final<-cbind(disaster_train, sent2$Sentiment_Score)
colnames(sent_final)<-c("id", "keyword", "location", "text", "target", "Sentiment_Score")
head(sent_final)
write.csv(sent_final, file="/Users/KnudseQ/Desktop/Sentiment.csv", row.names=FALSE)

t.test(sent_final$Sentiment_Score~sent_final$target, data=sent_final)

boxplot(sent_final$Sentiment_Score~sent_final$target, data=sent_final, notch=TRUE,
        col=(c("gold","darkgreen")),
        main="Sentiment Score for Real and Fake Disasters", xlab="Real (1) and Fake (0)", ylab="Sentiment Score")

view1 <- sent_final %>% arrange((sent_final$Sentiment_Score)) 

head(view1)

library(wesanderson)
library(ggplot2)
disaster_df$wordcount <- nchar(disaster_df$text)
summary(disaster_df$wordcount)
plotty <- ggplot(disaster_df, aes(x = as.factor(target), y = wordcount, fill = as.factor(target), group = target)) +
  theme_classic() +
  geom_boxplot() +
  labs(x = "Target (0) Fake (1) Real", y = "Tweet Word Count",
       title = "Distribution of Word Count for Real and Fake Disasters",
       subtitle = "Real Disasters Had Longer Tweets on Average")
plotty + scale_color_brewer(palette="Dark2")




a <- disaster_df %>% 
  count(target, sort = TRUE)


b <- disaster_df %>%
  group_by(target) %>%
  summarise(counts = n())

b$target <- ifelse(b$target == "1","Real Disaster",
                   ifelse(b$target == "0","Fake Disaster",0))


ggplot(b, aes(x = target, y = counts)) +
  geom_bar(fill = "#56B4E9", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3)+
  theme_classic() +
  labs(x = "Target", y = "Tweet Count",
       title = "Distribution of Real and Fake Disaster Tweets",
       subtitle = "Fake Disasters Have More Tweets in the Dataset")

a1 <- disaster_df %>% 
  count(location, sort = TRUE)


b1 <- disaster_df %>%
  group_by(location) %>%
  summarise(counts = n())%>% 
  top_n(n = 5, wt = location)

ggplot(b1, aes(x = location, y = counts)) +
  geom_bar(fill = "#56B4E9", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3)+
  theme_classic() +
  labs(x = "Target", y = "Tweet Count",
       title = "Distribution of Real and Fake Disaster Tweets",
       subtitle = "Fake Disasters Have More Tweets in the Dataset")


c <- disaster_df %>%
  group_by(location) %>%
  summarise(counts = n()) %>%
  arrange(counts, desc(counts))

d <- c %>%
  filter(counts >20) %>%
  arrange(counts, desc(counts))

library(ggplot2)
library(ggthemes)
ggplot(d[1:11,], 
       aes(x=location,  y=counts)) +
  geom_bar(stat="identity", fill='purple') + 
  coord_flip() + theme_classic() +  
  geom_text(aes(label=counts), colour="white",hjust=1.25, size=5.0)+
  labs(x = "Tweet Count", y = "Location",
       title = "Distribution of Location of Tweets",
       subtitle = "The Majority of Tweets Come from the US")


library(sentimentr)
sent <- sentiment(test_df$text)
sent1 <- as.data.frame(sent)
sent1
sent2 <- aggregate(sent1 [, 4], list(sent1$element_id), mean)
colnames(sent2)<-c("Id", "Sentiment_Score")
sent_final<-cbind(test_df, sent2$Sentiment_Score)
colnames(sent_final)<-c("id", "keyword", "location", "text", "Sentiment_Score")
head(sent_final)
write.csv(sent_final, file="/Users/KnudseQ/Desktop/Test Sentiment.csv", row.names=FALSE)

#Create text Corpus
corpus <- Corpus(VectorSource(disaster_df$text))
corpus[[1]][1]
disaster_df$target[1]
corpus[[2]][1]

# convert to lower case
corpus <- tm_map(corpus,PlainTextDocument)
corpus <-tm_map(corpus,tolower)
corpus[[1]][1]

#remove punctuation
corpus<- tm_map(corpus,removePunctuation)
corpus[[1]][1]

#remove stop words
corpus <- tm_map(corpus, removeWords,stopwords("english"))
corpus[[1]][1]

#Stemming 
corpus<- tm_map(corpus, stemDocument)
corpus[[1]][1]

corpus <- tm_map(corpus, lemmatize_strings)
# BUild a document Term matrix to look at frequencies of each word across all documents
frequency <- DocumentTermMatrix(corpus)
as.matrix(frequency)

#we can check the sparsity of this matrix by looking at summary below
frequency
as.matrix(removeSparseTerms(frequency,.995)) # this statement removes words that occur just once across all documents

sparse <- removeSparseTerms(frequency,.995)
sparse# 317 words appear atleast once across all documents

disaster_sparse <- as.data.frame(as.matrix(sparse))
colnames(disaster_sparse) <- make.names(colnames(disaster_sparse))

disaster_sparse$target<- disaster_df$target


train_tdm <- TermDocumentMatrix(corpus[1:7613], control= list(wordLengths= c(1, Inf)))
library(wordcloud)
library(RColorBrewer)
word.freq <- sort(rowSums(as.matrix((train_tdm))), decreasing= F)
pal <- brewer.pal(8, "Dark2")
wordcloud(words = names(word.freq), freq = word.freq, min.freq = 2, random.order = F, colors = pal, max.words = 150)

disaster_sparse <-cbind(disaster_sparse, sent2$Sentiment_Score)
write.csv(sent_final, file="/Users/KnudseQ/Desktop/sparse.csv", row.names=FALSE)

disaster_sparse <-cbind(disaster_sparse, disaster_df)
disaster_sparse$location
#Split combined data frame to train and test
set.seed(34)
ind <- sample(2, nrow(disaster_sparse), replace =T, prob = c(0.7, 0.3))
train <- disaster_sparse[ind==1,]
test <- disaster_sparse[ind==2,]
library(caTools)
library(cluster)
library(dplyr)
library(readr)
library(cluster)
library(factoextra)
km_output <- kmeans(disaster_sparse, centers = 2, nstart = 25, iter.max = 100, algorithm = "Hartigan-Wong")
fviz_cluster(km_output, data = disaster_sparse)
#split <- sample.split(train$target,SplitRatio = 0.7)
#build train validation and test validation sets using train
#train_v <- subset(train, split==TRUE)
#test_v<- subset(train, split==FALSE)

library(randomForest)

#convert target variable to factor
train$target <- as.factor(train$target)
test$target <- as.factor(test$target)


 

svm <- train(target~.,data = train, method = "svmLinear", metric ="Accuracy", trControl = trainControl(method="cv", number=3), preProcess = c("center","scale"))
predict <- predict(svm,newdata = test) #nrow(test_v) = 2284
cm_svm <- confusionMatrix(predict,test$target)

train$sentiment.score <- train$`sent2$Sentiment_Score`
make.names(colnames(train))

model <- randomForest(target~.,data = train, na.action=na.exclude) #nrow(train_v) = 5329
predict <- predict(model,newdata = test) #nrow(test_v) = 2284
confusionMatrix(predict,test$target)

varImpPlot(model,type=2)

library(e1071)
svm<- svm(target~.,data = train)
predict <- predict(svm,newdata = test) #nrow(test_v) = 2284
confusionMatrix(predict,test$target)




nb<- naiveBayes(target~.,data = train, laplace = 1, na.action = na.pass)
predict <- predict(nb,newdata = test) #nrow(test_v) = 2284
cm_nb <-confusionMatrix(predict,test$target)


#predict using model on the test data set
final_predict <- predict(model, test) #nrow(test) 3263
submission<- read.csv("sample_submission.csv")
write.csv(submission, "/kaggle/working/submission.csv", row.names = F)

## Saving data

# If you save any files or images, these will be put in the "output" directory. You 
# can see the output directory by committing and running your kernel (using the 
# Commit & Run button) and then checking out the compiled version of your kernel.