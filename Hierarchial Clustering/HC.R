# Hierarchial clustering

# Importing the mall dataset
dataset = read.csv('Mall_Customers.csv')
x = dataset[4:5]

# Using dendrogram to find optimal number of clusters
dendrogram = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting hierarchial clustering to mall dataset
hc = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
library(cluster)
clusplot(x, 
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters of Clients"),
         xlab = "Annual Income",
         ylab = "Spending Score")