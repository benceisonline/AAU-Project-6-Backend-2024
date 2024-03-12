#!/usr/bin/env python
# coding: utf-8

# ## Shell

# In[31]:


get_ipython().run_line_magic('pip', 'install pytorch_lightning')
get_ipython().run_line_magic('pip', 'install torchmetrics')
get_ipython().run_line_magic('pip', 'install --upgrade tensorboard')
get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install nbconvert')


# ## Imports

# In[32]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import os
from collections import Counter


# ## Import for TensorBoard

# In[33]:


from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("tb_logs", name="my_model")


# ## Data preprocessing

# In[34]:


raw_behaviour = pd.read_csv(
    "./MIND_small/behaviors.tsv",
    sep="\t",
    names=["impressionId","userId","timestamp","click_history","impressions"])

print(f"The dataset consist of {len(raw_behaviour)} number of interactions.")
raw_behaviour.head()


# ## Indexize users

# In[35]:


# Indexize users
unique_userIds = raw_behaviour['userId'].unique()
# Allocate a unique index for each user, but let the zeroth index be a UNK index:
ind2user = {idx +1: itemid for idx, itemid in enumerate(unique_userIds)}
user2ind = {itemid : idx for idx, itemid in ind2user.items()}
print(f"We have {len(user2ind)} unique users in the dataset")

# Create a new column with userIdx:
raw_behaviour['userIdx'] = raw_behaviour['userId'].map(lambda x: user2ind.get(x,0))


# ## Load article data

# In[36]:


news = pd.read_csv(
    "./MIND_small/news.tsv",
    sep="\t",
    names=["itemId","category","subcategory","title","abstract","url","title_entities","abstract_entities"])
news.head(2)
# Build index of items
ind2item = {idx +1: itemid for idx, itemid in enumerate(news['itemId'].values)}
item2ind = {itemid : idx for idx, itemid in ind2item.items()}

news.head()


# ## Indexize click history field

# In[37]:


# Indexize click history field
def process_click_history(s):
    list_of_strings = str(s).split(" ")
    return [item2ind.get(l, 0) for l in list_of_strings]

raw_behaviour['click_history_idx'] = raw_behaviour.click_history.map(lambda s:  process_click_history(s))
raw_behaviour.head()


# ## Collect one click and no click impressions

# In[38]:


# collect one click and one no-click from impressions:
def process_impression(s):
    list_of_strings = s.split(" ")
    itemid_rel_tuple = [l.split("-") for l in list_of_strings]
    noclicks = []
    for entry in itemid_rel_tuple:
        if entry[1] =='0':
            noclicks.append(entry[0])
        if entry[1] =='1':
            click = entry[0]
    return noclicks, click

raw_behaviour['noclicks'], raw_behaviour['click'] = zip(*raw_behaviour['impressions'].map(process_impression))
# We can then indexize these two new columns:
raw_behaviour['noclicks'] = raw_behaviour['noclicks'].map(lambda list_of_strings: [item2ind.get(l, 0) for l in list_of_strings])
raw_behaviour['click'] = raw_behaviour['click'].map(lambda x: item2ind.get(x,0))


# In[39]:


raw_behaviour.head()


# ## Convert timestamp value to hours since epoch

# In[40]:


raw_behaviour['epochhrs'] = pd.to_datetime(raw_behaviour['timestamp']).values.astype(np.int64)/(1e6)/1000/3600
raw_behaviour['epochhrs'] = raw_behaviour['epochhrs'].round()
raw_behaviour[['click','epochhrs']].groupby("click").min("epochhrs").reset_index()


# In[41]:


raw_behaviour


# ## Modeling

# In[42]:


raw_behaviour['noclick'] = raw_behaviour['noclicks'].map(lambda x : x[0])
behaviour = raw_behaviour[['epochhrs','userIdx','click_history_idx','noclick','click']]
behaviour.head()


# In[43]:


# Let us use the last 10pct of the data as our validation data:
test_time_th = behaviour['epochhrs'].quantile(0.9)
train = behaviour[behaviour['epochhrs']< test_time_th]
valid =  behaviour[behaviour['epochhrs']>= test_time_th]


# In[44]:


class MindDataset(Dataset):
    # A fairly simple torch dataset module that can take a pandas dataframe (as above),
    # and convert the relevant fields into a dictionary of arrays that can be used in a dataloader
    def __init__(self, df):
        # Create a dictionary of tensors out of the dataframe
        self.data = {
            'userIdx' : torch.tensor(df.userIdx.values),
            'click' : torch.tensor(df.click.values),
            'noclick' : torch.tensor(df.noclick.values)
        }
    def __len__(self):
        return len(self.data['userIdx'])
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


# In[45]:


# Build datasets and dataloaders of train and validation dataframes:
bs = 1024
ds_train = MindDataset(train)
train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
ds_valid = MindDataset(valid)
valid_loader = DataLoader(ds_valid, batch_size=bs, shuffle=False)

batch = next(iter(train_loader))


# In[46]:


batch["noclick"]


# ## Model
# ### Framework
# 
# We will use pytorch-lightning to define and train our model. It is a high-level framework (similar to fastAI) but with a slightly different way of defining things. It is my personal go-to framework and is very flexible. For more information, see https://pytorch-lightning.readthedocs.io/.
# The model
# 
# We assume that each interaction goes as follow: the user is presented with two items: the click and no-click item. After the user reviewed both items, she will choose the most relevant one. This can be modeled as a categorical distirbution with two options (yes, you could do binomial). There is a loss function in pytorch for this already, called the F.cross_entropy that we will use.
# 

# In[47]:


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

# Instantiate the model
mf_model = NewsMF(num_users=len(ind2user)+1, num_items=len(ind2item)+1)


# In[48]:


trainer = pl.Trainer(max_epochs=10,logger=logger)
trainer.fit(model=mf_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


# In[49]:


logs = trainer.logged_metrics

# Print or inspect the logs
print("Training and validation logs:", logs)


# ## Sense check

# In[50]:


USER_ID = 2350 # Random user id


# ## Suggested items

# In[51]:


# Create item_ids and user ids list
item_id = list(ind2item.keys())
userIdx =  [USER_ID]*len(item_id)

preditions = mf_model.forward(torch.IntTensor(userIdx), torch.IntTensor(item_id))

# Select top 10 argmax
top_index = torch.topk(preditions.flatten(), 10).indices

# Filter for top 10 suggested items
filters = [ind2item[ix.item()] for ix in top_index]
news[news["itemId"].isin(filters)]


# ## Historical items

# In[52]:


click_ids = behaviour[behaviour["userIdx"]==USER_ID]["click"].values
ll = lambda x: "N"+str(x)

click_ids = [ll(each) for each in click_ids]

news[news["itemId"].isin(click_ids)]


# ## Tensorboard

# In[67]:


# Load the extension and start TensorBoard

get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir tb_logs')


# ## Saving the model

# In[54]:


# Specify the relative directory path
relative_directory = "Saved_Model/"

# Create the full directory path
directory_path = os.path.join(relative_directory)

# Create the directory if it does not exist
os.makedirs(directory_path, exist_ok=True)

# Save the state dictionary of the model to the specified directory
model_save_path = os.path.join(directory_path, "collaborative_filtering_model.pth")
torch.save(mf_model.state_dict(), model_save_path)


# ## Loading and running the model

# In[55]:


# Load the state dictionary from the specified directory
loaded_model = NewsMF(num_users=len(ind2user)+1, num_items=len(ind2item)+1)

# Use a relative path when loading the model
model_load_path = os.path.join("Saved_Model", "collaborative_filtering_model.pth")
loaded_model.load_state_dict(torch.load(model_load_path))


# In[56]:


# Specify the user ID for prediction
USER_ID = 1234

# Create item_ids and user ids list
item_id = list(ind2item.keys())
userIdx = [USER_ID] * len(item_id)

# Convert lists to PyTorch tensors
user_tensor = torch.IntTensor(userIdx)
item_tensor = torch.IntTensor(item_id)

# Forward pass to get predictions
predictions = loaded_model.forward(user_tensor, item_tensor)

# Select top 10 indices
top_indices = torch.topk(predictions.flatten(), 10).indices

# Get corresponding item IDs
top_item_ids = [ind2item[ix.item()] for ix in top_indices]

# Filter for top 10 suggested items
recommended_items = news[news["itemId"].isin(top_item_ids)]

# Display the recommended items
print(recommended_items)


# #Convert to Python Script (not needed right now but keep as utility)

# In[57]:


get_ipython().system('python -m nbconvert --to script collab.ipynb')


# ## Get random user id

# In[58]:


random_user_index = np.random.randint(0, len(raw_behaviour))
random_user_id = raw_behaviour.iloc[random_user_index]['userId']

print(f"Randomly selected user ID: {random_user_id}")

