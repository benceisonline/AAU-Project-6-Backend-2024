#!/usr/bin/env python
# coding: utf-8

# ### Shell

# In[22]:


get_ipython().run_line_magic('pip', 'install pytorch_lightning')
get_ipython().run_line_magic('pip', 'install torchmetrics')
get_ipython().run_line_magic('pip', 'install --upgrade tensorboard')
get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install nbconvert')


# ### Imports

# In[23]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import os
from collections import Counter


# ### Import for TensorBoard

# In[24]:


from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("tb_logs", name="my_model")


# ### Data Preprocessing

# In[25]:


# Load EBNeRD behaviors dataset for both train and validation
train_behaviour = pd.read_parquet("./ebnerd_small/train/behaviors.parquet")
valid_behaviour = pd.read_parquet("./ebnerd_small/validation/behaviors.parquet")
behaviors = pd.concat([train_behaviour, valid_behaviour], ignore_index=True)

behaviors.head()


# In[26]:


# Load EBNeRD history dataset for both train and validation
train_history = pd.read_parquet("./ebnerd_small/train/history.parquet")
valid_history = pd.read_parquet("./ebnerd_small/validation/history.parquet")
history = pd.concat([train_history, valid_history], ignore_index=True)

history.head()


# In[27]:


# Load EBNeRD news dataset
news = pd.read_parquet("./ebnerd_small/articles.parquet")

news.head()


# ### Join history and behaviour tables

# In[28]:


# Left join on 'user_id'
behaviour_history_merged= pd.merge(behaviors, history, on='user_id', how='left')

# Display the merged data
behaviour_history_merged.head()


# ### Generate binary labels

# In[29]:


# Function to create binary labels column
def create_binary_labels_column(df):
    # Define the column names
    clicked_col = "article_ids_clicked"
    inview_col = "article_ids_inview"
    labels_col = "labels"

    # Create a new column with binary labels
    df[labels_col] = df.apply(lambda row: [1 if article_id in row[clicked_col] else 0 for article_id in row[inview_col]], axis=1)

    # Shuffle the data
    df = df.sample(frac=1, random_state=123)

    # Add a column with the length of the labels list
    df[labels_col + "_len"] = df[labels_col].apply(len)

    return df

# Apply the function to your merged dataset
behaviour_history_merged = create_binary_labels_column(behaviour_history_merged)

# Display the updated dataset
behaviour_history_merged.head()


# In[30]:


# Indexize users for the new dataset
unique_user_ids = behaviour_history_merged['user_id'].unique()
user2ind = {itemid: idx for idx, itemid in enumerate(unique_user_ids)}
ind2user = {idx +1: itemid for idx, itemid in enumerate(unique_user_ids)}
behaviour_history_merged['userIdx'] = behaviour_history_merged['user_id'].map(lambda x: user2ind.get(x, 0))
print(f"We have {len(user2ind)} unique users in the dataset")


# In[31]:


# Indexize articles for the new dataset
unique_article_ids = behaviour_history_merged['article_id'].unique()
article2ind = {itemid: idx for idx, itemid in enumerate(unique_article_ids)}
ind2article = {idx +1: itemid for idx, itemid in enumerate(unique_article_ids)}
behaviour_history_merged['articleIdx'] = behaviour_history_merged['article_id'].map(lambda x: article2ind.get(x, 0))
print(f"We have {len(article2ind)} unique articles in the dataset")


# In[32]:


# Split data into train and validation
test_time_threshold = behaviour_history_merged['impression_time'].quantile(0.9)
train_data = behaviour_history_merged[behaviour_history_merged['impression_time'] < test_time_threshold]
valid_data = behaviour_history_merged[behaviour_history_merged['impression_time'] >= test_time_threshold]


# In[33]:


class EBNeRDMindDataset(Dataset):
    def __init__(self, df):
        self.data = {
            'userIdx': torch.tensor(df.userIdx.values),
            'articleIdx': torch.tensor(df.articleIdx.values),
            'labels': torch.tensor([item for sublist in df.labels for item in sublist], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.data['userIdx'])

    def __getitem__(self, idx):
        return {
            'userIdx': self.data['userIdx'][idx],
            'articleIdx': self.data['articleIdx'][idx],
            'click': self.data['labels'][idx].long(),
            'noclick': 1 - self.data['labels'][idx].long(),
        }


# In[34]:


# Build datasets and dataloaders for train and validation dataframes
bs = 1024
ds_train = EBNeRDMindDataset(train_data)
train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
ds_valid = EBNeRDMindDataset(valid_data)
valid_loader = DataLoader(ds_valid, batch_size=bs, shuffle=False)


# ### Model

# In[35]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryAUROC

class NewsMF(pl.LightningModule):
    def __init__(self, num_users, num_items, dim=10):
        super().__init__()
        self.dim = dim
        self.useremb = nn.Embedding(num_embeddings=num_users, embedding_dim=dim)
        self.itememb = nn.Embedding(num_embeddings=num_items, embedding_dim=dim)
        
        # BinaryF1Score metric
        self.f1_metric = BinaryF1Score()
        self.train_step_f1_outputs = []
        self.validation_step_f1_outputs = []

        # BinaryAUROC metric
        self.binary_auroc = BinaryAUROC()
        self.train_step_auroc_outputs = []
        self.validation_step_auroc_outputs = []


    def forward(self, user, item):
        batch_size = user.size(0)
        uservec = self.useremb(user)
        itemvec = self.itememb(item)

        score = (uservec * itemvec).sum(-1).unsqueeze(-1)

        return score

    def training_step(self, batch, batch_idx):
        batch_size = batch['userIdx'].size(0)

        # Compute loss as cross entropy (categorical distribution between the clicked and the no-clicked item)
        score_click = self.forward(batch['userIdx'], batch['click'])
        score_noclick = self.forward(batch['userIdx'], batch['noclick'])

        loss = F.cross_entropy(input=torch.cat((score_click, score_noclick), dim=1),
                               target=torch.zeros(batch_size, device=score_click.device).long())
        
        # Compute F1-score
        f1_click = self.f1_metric(score_click.squeeze(), torch.ones_like(batch['click']))
        f1_noclick = self.f1_metric(score_noclick.squeeze(), torch.zeros_like(batch['noclick']))

        # Average F1-scores
        f1 = (f1_click + f1_noclick) / 2.0

        self.train_step_f1_outputs.append(f1)

        # Calculate Binary AUROC
        binary_auroc_score = self.binary_auroc(torch.cat((score_click, score_noclick), dim=1),
                                                torch.cat((torch.ones_like(batch['click']),
                                                           torch.zeros_like(batch['noclick'])))
                                               )
        
        self.train_step_auroc_outputs.append(binary_auroc_score)

        # Log metrics to TensorBoard
        self.log('train_loss', loss)
        self.log('train_f1', f1)
        self.log('train_auroc', binary_auroc_score)

        return {'loss': loss, 'f1': f1, 'auroc': binary_auroc_score}
    
    def on_train_epoch_end(self):
        epoch_average_f1 = torch.stack(self.train_step_f1_outputs).mean()
        print(f'Epoch {self.current_epoch}: Training F1 Score: {epoch_average_f1.item()}')
        self.log("train_epoch_average_f1", epoch_average_f1)
        self.train_step_f1_outputs.clear()  # free memory

        epoch_average_auroc = torch.stack(self.train_step_auroc_outputs).mean()
        print(f'Epoch {self.current_epoch}: Training AUROC Score: {epoch_average_auroc.item()}')
        self.log("train_epoch_average_auroc", epoch_average_auroc)
        self.validation_step_auroc_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        # Compute loss as cross-entropy (categorical distribution between clicked and non-clicked items)
        score_click = self.forward(batch['userIdx'], batch['click'])
        score_noclick = self.forward(batch['userIdx'], batch['noclick'])

        loss = F.cross_entropy(input=torch.cat((score_click, score_noclick), dim=1),
                            target=torch.zeros(batch['userIdx'].size(0), device=score_click.device).long())
        
        # F1 Score
        f1_click = self.f1_metric(score_click.squeeze(), torch.ones_like(batch['click']))
        f1_noclick = self.f1_metric(score_noclick.squeeze(), torch.zeros_like(batch['noclick']))
        f1 = (f1_click + f1_noclick) / 2.0 # Average F1-scores

        self.validation_step_f1_outputs.append(f1)

        # Calculate Binary AUROC
        binary_auroc_score = self.binary_auroc(torch.cat((score_click, score_noclick), dim=1),
                                                torch.cat((torch.ones_like(batch['click']),
                                                           torch.zeros_like(batch['noclick'])))
                                               )
        
        self.validation_step_auroc_outputs.append(binary_auroc_score)

        # Log metrics to TensorBoard
        self.log('val_loss', loss)
        self.log('val_f1', f1)
        self.log('val_auroc', binary_auroc_score)
                
        return {'loss': loss, 'f1': f1, 'auroc': binary_auroc_score}

    def on_validation_epoch_end(self):
        epoch_average_f1 = torch.stack(self.validation_step_f1_outputs).mean()
        print(f'Epoch {self.current_epoch}: Validation F1 Score: {epoch_average_f1.item()}')
        self.log("validation_epoch_average_f1", epoch_average_f1)
        self.validation_step_f1_outputs.clear()  # free memory

        epoch_average_auroc = torch.stack(self.validation_step_auroc_outputs).mean()
        print(f'Epoch {self.current_epoch}: Validation AUROC Score: {epoch_average_auroc.item()}')
        self.log("validation_epoch_average_auroc", epoch_average_auroc)
        self.validation_step_auroc_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# In[36]:


ebnerd_model = NewsMF(num_users=len(user2ind) + 1, num_items=len(article2ind) + 1)


# In[37]:


# Instantiate the trainer
trainer = pl.Trainer(max_epochs=10, logger=logger)

# Train the model
trainer.fit(model=ebnerd_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


# In[38]:


logs = trainer.logged_metrics

# Print or inspect the logs
print("Training and validation logs:", logs)


# ### Prediction test

# In[39]:


USER_ID = 2350 # Random user id
# Create item_ids and user ids list
item_id = list(ind2article.keys())
userIdx =  [USER_ID]*len(item_id)

preditions = ebnerd_model.forward(torch.IntTensor(userIdx), torch.IntTensor(item_id))

# Select top 10 argmax
top_index = torch.topk(preditions.flatten(), 10).indices

# Filter for top 10 suggested items
filters = [ind2article[ix.item()] for ix in top_index]
news[news["article_id"].isin(filters)]


# ### Model Save

# In[40]:


# Specify the relative directory path
relative_directory = "Saved_Model/"

# Create the full directory path
directory_path = os.path.join(relative_directory)

# Create the directory if it does not exist
os.makedirs(directory_path, exist_ok=True)

# Save the state dictionary of the model to the specified directory
model_save_path = os.path.join(directory_path, "EBNERD_collaborative_filtering_model.pth")
torch.save(ebnerd_model.state_dict(), model_save_path)


# ### Model Load

# In[41]:


# Load the state dictionary from the specified directory
loaded_model = NewsMF(num_users=len(ind2user)+1, num_items=len(ind2article)+1)

# Use a relative path when loading the model
model_load_path = os.path.join("Saved_Model", "EBNERD_collaborative_filtering_model.pth")
loaded_model.load_state_dict(torch.load(model_load_path))


# ### Loaded Model Single Prediciton

# In[42]:


# Specify the user ID for prediction
USER_ID = 1234
PREDICTION_COUNT = 10

# Create item_ids and user ids list
article_id = list(ind2article.keys())
userIdx = [USER_ID] * len(article_id)

# Convert lists to PyTorch tensors
user_tensor = torch.IntTensor(userIdx)
item_tensor = torch.IntTensor(article_id)

# Forward pass to get predictions
predictions = loaded_model.forward(user_tensor, item_tensor)

# Select top 10 indices
top_indices = torch.topk(predictions.flatten(), PREDICTION_COUNT).indices

# Get corresponding item IDs
top_item_ids = [ind2article[ix.item()] for ix in top_indices]

# Filter for top 10 suggested items
recommended_items = news[news["article_id"].isin(top_item_ids)]

# Display the recommended items
print(recommended_items)


# ### Tensorboard

# In[43]:


# Load the extension and start TensorBoard
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir tb_logs')


# ### Convert to Python Script (not needed right now but keep as utility)

# In[44]:


get_ipython().system('python -m nbconvert --to script EBNERD_Notebook.ipynb')


# ### Get random user id

# In[45]:


random_user_index = np.random.randint(0, len(behaviors))
random_user_id = behaviors.iloc[random_user_index]['user_id']

print(f"Randomly selected user ID: {random_user_id}")

