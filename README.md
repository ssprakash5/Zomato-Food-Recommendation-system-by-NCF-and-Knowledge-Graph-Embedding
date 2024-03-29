# Zomato-Food-Recommendation-system-by-NCF-and-Knowledge-Graph-Embedding
![zoma](https://github.com/ssprakash5/Zomato-Food-Recommendation-system-by-NCF-and-Knowledge-Graph-Embedding/assets/154003057/bee75a49-0e9e-4da7-8233-a0c6fc722103)
![Zomato Food Recommendation for Each number](https://github.com/ssprakash5/Zomato-Food-Recommendation-system-by-NCF-and-Knowledge-Graph-Embedding/assets/154003057/3ea4a397-71c7-4cc4-b3a7-a87196a0b415)


## Project Overview

This project aims to enhance Zomato's food recommendation system by incorporating advanced techniques such as Neural Collaborative Filtering (NCF) and Knowledge Graph Embeddings. The goal is to provide users with more personalized and accurate food recommendations based on their preferences, past behavior, and the relationships between different food items. The model trained with Zomato Bangalore Restaurants dataset(https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)

## Skills and Technologies Utilized

- Python (PyTorch)
- Data Preprocessing
- Collaborative Filtering
- Neural Networks
- Knowledge Graph Embeddings
- Recommendation Systems

## Project Objectives

### Data Collection and Preprocessing

- Gathered https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants dataset which consists of Zomato's historical user interaction data, including user reviews, ratings, and order history.
- Preprocess the data by handling missing values, removing duplicates, and encoding categorical variables.
![image](https://github.com/ssprakash5/Zomato-Food-Recommendation-system-by-NCF-and-Knowledge-Graph-Embedding/assets/154003057/b1900e37-5534-4b84-ac7b-b9c283104e16)

### Knowledge Graph Construction

- Construct a knowledge graph using available data, representing relationships between food items, cuisines, restaurants, and user preferences.
- Define and incorporate ontologies for entities in the knowledge graph.
![knowledge_graph](https://github.com/ssprakash5/Zomato-Food-Recommendation-system-by-NCF-and-Knowledge-Graph-Embedding/assets/154003057/fb10c426-8c70-4d89-9921-8f9fdc2bd62a)

### Neural Collaborative Filtering (NCF)

- Implement Neural Collaborative Filtering using neural networks to capture user-item interactions.
- Train the NCF model on the user interaction data to learn user and item embeddings.
![NCF](https://github.com/ssprakash5/Zomato-Food-Recommendation-system-by-NCF-and-Knowledge-Graph-Embedding/assets/154003057/58dd2f51-e51c-4841-9893-99cfdcf171ca)

### Knowledge Graph Embeddings

- Apply techniques such as TransE or DistMult to embed knowledge graph information into a vector space.
- Integrate the learned embeddings with the NCF model to incorporate semantic relationships between food items.

### Model Training and Evaluation

- Split the dataset into training and testing sets.
- Train the integrated model on the training set and evaluate its performance on the testing set using relevant metrics such as precision, recall, and F1-score.

### Hyperparameter Tuning

- Fine-tune the hyperparameters of both the NCF model and the knowledge graph embedding model to optimize performance.

### Recommendation Model
- The Recommendation system gives the top 10 dishes/cuisines with Restaurant names and cost of two for each location in Bangalore.
## Expected Outcomes

- A more accurate and personalized food recommendation system for Zomato users.
- Improved user engagement and satisfaction with the Zomato platform.
- Insights into the impact of knowledge graph embeddings on recommendation system performance.

## Impact

- Will Enhance user experience leading to increased customer retention.
- Will Increase revenue for Zomato through improved user engagement and satisfaction.

