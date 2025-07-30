import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split

DATA_FILENAME = "data/slice_localization_data.csv"

dataframe = pd.read_csv(DATA_FILENAME)
dataframe = dataframe.drop(['patientId'], axis=1)
dataframe.head()


num_rows = len(dataframe)
print(num_rows)


num_cols = len(dataframe.columns)
print(num_cols)


input_cols = list(dataframe.columns.values)
input_cols.pop()


output_cols = ['reference']



# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# scaled_df = scaler.fit_transform(dataframe)
# pca = PCA(0.95)
# pca_vectors = pca.fit_transform(scaled_df )
# for index, var in enumerate(pca.explained_variance_ratio_):
#     print("Explained Variance ratio by Principal Component ", (index+1), " : ", var)


def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array


inputs_array, targets_array = dataframe_to_arrays(dataframe)

inputs = torch.from_numpy(inputs_array).type(torch.float32)
targets = torch.from_numpy(targets_array).type(torch.float32)

dataset = TensorDataset(inputs, targets)
batch_size = 8

val_percent = 0.2 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset, [train_size, val_size]) # Use the random_split function to split dataset into 2 parts of the desired length

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = len(input_cols)
output_size = len(output_cols)
# print("input_size", input_size)
# print("output_size", output_size)


class CTslicesModel(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.PReLU(),
            nn.Linear(1000, 500),
            nn.PReLU(),
            nn.Linear(500, 250),
            nn.PReLU(),
            nn.Linear(250, 100),
            nn.PReLU(),
            nn.Linear(100, output_size)   
         )
         '''
        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size)
        )
        
    def forward(self, xb):
        out = self.linear(xb)            
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets)                       
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets) 

        # Calculate RMSE
        criterion = nn.MSELoss()
        rmse= torch.sqrt(criterion(out, targets))                        
        return {'val_loss': loss.detach(), 'val_rmse': rmse}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_rmses = [x['val_rmse'] for x in outputs]
        epoch_rmse = torch.stack(batch_rmses).mean()      # Combine RMSEs
        return {'val_loss': epoch_loss.item(), 'val_rmse': epoch_rmse.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 2 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss'], result['val_rmse']))


# input 384 output 1
model = CTslicesModel()

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


epochs = 50
lr = 0.01
history1 = fit(epochs, lr, model, train_loader, val_loader)

result = evaluate(model, val_loader) # Use the the evaluate function
print(result)



def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');

def plot_rmse(history):
    rmses = [x['val_rmse'] for x in history]
    plt.plot(rmses , '-x')
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. No. of epochs');



# plot_losses(history1)
# plot_rmse(history1)


model1 = torch.load('slice.pth')
model1.eval()

example_input = torch.rand(1, 384)
scripted_model = torch.jit.trace(model1, example_input)
torch.jit.save(scripted_model, 'slice.pt')


def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    print("res input", input)
    predictions = model(input) 
    print(predictions)              # fill this
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)

input, target = val_ds[0]
predict_single(input, target, model1)

print(input_size)