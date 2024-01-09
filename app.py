import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import pairwise_distance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load your Zomato DataFrame (replace 'df' with your actual DataFrame)
# Ensure 'name', 'dish_liked', and 'rate' are your column names
df = pd.read_csv('/content/zomato_cleaned.csv')  # Update with your data file

# Collaborative Filtering Model

# Assuming 'name', 'dish_liked', and 'rate' are your column names
ncf_data = df[['name', 'dish_liked', 'rate']]

# Encode 'name' and 'dish_liked' to integer indices
ncf_data['user_id'] = ncf_data['name'].astype('category').cat.codes
ncf_data['item_id'] = ncf_data['dish_liked'].astype('category').cat.codes

# Split the data into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(ncf_data, test_size=0.2, random_state=42)

# Define the input layers for user and item
user_input = torch.tensor(test_data['user_id'].values, dtype=torch.long)
item_input = torch.tensor(test_data['item_id'].values, dtype=torch.long)

# Load your pre-trained collaborative filtering model
# You can use the 'ncf_model' defined in the collaborative filtering section

# Knowledge Graph Embeddings Model

# Extract relevant columns for constructing the knowledge graph
graph_data = df[['name', 'cuisines', 'dish_liked']]

# Create triples for the knowledge graph (restaurant, serves, cuisine) and (restaurant, likes, dish)
triples = []
for index, row in graph_data.iterrows():
    restaurant = row['name']
    cuisines = row['cuisines'].split(', ') if pd.notnull(row['cuisines']) else []
    dish_liked = row['dish_liked'].split(', ') if pd.notnull(row['dish_liked']) else []

    # Add triples for cuisine
    for cuisine in cuisines:
        triples.append((restaurant, 'serves', cuisine))

    # Add triples for dish
    for dish in dish_liked:
        triples.append((restaurant, 'likes', dish))

# Convert triples to PyTorch tensors
entities = set(triple[0] for triple in triples) | set(triple[2] for triple in triples)
relations = set(triple[1] for triple in triples)
num_entities = len(entities)
num_relations = len(relations)

entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
relation_to_id = {relation: idx for idx, relation in enumerate(relations)}

triples_ids = [(entity_to_id[h], relation_to_id[r], entity_to_id[t]) for h, r, t in triples if h in entity_to_id and t in entity_to_id]
triples_tensor = torch.tensor(triples_ids, dtype=torch.long)

# Split the data into training and testing sets
train_triples, test_triples = train_test_split(triples_tensor.numpy(), test_size=0.2, random_state=42)
train_triples = torch.tensor(train_triples, dtype=torch.long)
test_triples = torch.tensor(test_triples, dtype=torch.long)

# Define the HolE model
class HolE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(HolE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, head, relation, tail):
        head_embedding = self.entity_embeddings(head)
        relation_embedding = self.relation_embeddings(relation)
        tail_embedding = self.entity_embeddings(tail)

        # Ensure compatibility by reshaping head_embedding and tail_embedding
        head_embedding = head_embedding.view(-1, 1, self.embedding_dim)
        tail_embedding = tail_embedding.view(-1, 1, self.embedding_dim)

        correlation = torch.sum(head_embedding * tail_embedding, dim=2)
        return torch.sigmoid(correlation * relation_embedding).squeeze()

# Instantiate the HolE model
embedding_dim = 50  # Adjust the embedding dimension as needed
hole_model = HolE(num_entities, num_relations, embedding_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(hole_model.parameters(), lr=0.001)  # Adjust the learning rate as needed

# Train the HolE model
num_epochs = 50  # Adjust the number of epochs as needed
batch_size = 64  # Adjust batch size as needed

train_dataset = TensorDataset(train_triples[:, 0], train_triples[:, 1], train_triples[:, 2])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        head, relation, tail = batch
        output = hole_model(head, relation, tail)
        loss = criterion(output, torch.ones_like(output))
        loss.backward()
        optimizer.step()

# Streamlit App

def get_ncf_recommendations(user_id, model, data_loader):
    # Placeholder for collaborative filtering recommendations
    # Replace with actual collaborative filtering logic

    # Encode user_id and item_ids
    user_input = torch.tensor([user_id], dtype=torch.long)
    item_input = torch.arange(len(data_loader.get_food_items()), dtype=torch.long)

    # Predictions using the NCF model
    predictions = model.predict([user_input, item_input])

    # Get top 5 recommended item indices
    top_indices = np.argsort(predictions.flatten())[-5:][::-1]

    # Map indices to actual food items
    recommendations = [data_loader.get_food_items()[i] for i in top_indices]

    return recommendations
def get_hole_recommendations(user_input, model, entity_to_id, id_to_entity, num_recommendations=5):
    # Placeholder for knowledge graph embeddings recommendations
    # Replace with actual knowledge graph embeddings logic

    # Encode the user_input
    user_id = entity_to_id.get(user_input, -1)
    if user_id == -1:
        return ["No recommendations available for the given user."]

    # Create tensor input for the model
    user_input_tensor = torch.tensor([user_id], dtype=torch.long)
    all_item_ids = torch.arange(len(id_to_entity), dtype=torch.long)

    # Predictions using the HolE model
    predictions = model.predict([user_input_tensor, all_item_ids])

    # Get top recommended item indices
    top_indices = torch.argsort(predictions.flatten(), descending=True)[:num_recommendations]

    # Map indices to actual items
    recommendations = [id_to_entity[i.item()] for i in top_indices]

    return recommendations

def main():
    st.title("Zomato Food Recommendation App")

    # Get user input
    user_input = st.text_input("Enter your user ID:", "")

    if st.button("Get Collaborative Filtering Recommendations"):
        # Generate collaborative filtering recommendations based on user ID
        recommendations_ncf = get_ncf_recommendations(int(user_input), ncf_model, data_loader)

        # Display collaborative filtering recommendations
        st.subheader("Collaborative Filtering Recommendations:")
        for i, recommendation in enumerate(recommendations_ncf, 1):
            st.write(f"{i}. {recommendation}")

    if st.button("Get Knowledge Graph Embeddings Recommendations"):
        # Generate knowledge graph embeddings recommendations based on user input
        recommendations_hole = get_hole_recommendations(user_input, hole_model)

        # Display knowledge graph embeddings recommendations
        st.subheader("Knowledge Graph Embeddings Recommendations:")
        for i, recommendation in enumerate(recommendations_hole, 1):
            st.write(f"{i}. {recommendation}")

if __name__ == "__main__":
    main()
