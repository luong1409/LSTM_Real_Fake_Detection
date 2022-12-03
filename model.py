import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from loguru import logger

import os


class RealFakeDetection(nn.Module):
    
    def __init__(self, max_features, dimension=128):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=max_features, embedding_dim=300)
        self.dimension = dimension
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.drop = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(
            in_features=2*dimension,
            out_features=1
        )
    
    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        
        packed_input = pack_padded_sequence(
            input=text_emb, 
            lengths=text_len,
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(
            sequence=packed_output,
            batch_first=True
        )
        
        # for getting the final dept-wise and time-wise
        out_forward = output[:, max(text_len) - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat(tensors=(out_forward, out_reverse),
                                dim=1)
        # current shape is: (batch_size, 2*self.dimension)
        text_fea = self.drop(out_reduced)
        # -> shape: (batch_size, 1)
        # because fully connected layer have out_features is `1`
        text_fea = self.fc(text_fea)
        
        # squeeze with dim=1 will return flatten tensor.
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)
        
        return text_out


def save_checkpoint(save_path, model:RealFakeDetection, optimizer:optim.Optimizer, valid_loss):
    
    if save_path == None:
        logger.warning("save_path is None. So save_path is set to ==> 'models/lstm.pt'")
        os.makedirs("models/", exist_ok=True)
        save_path = "models/lstm.pt"
    
    folder_path = os.path.split(save_path)[0]
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path,)
    
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'valid_loss': valid_loss
    }
    
    torch.save(obj=state_dict, f=save_path)
    logger.info(f"Model saved to ==> {save_path}")


def load_checkpoint(load_path, model:RealFakeDetection, optimizer:optim.Optimizer, device):
    if load_path == None:
        logger.warning("load_path is None. So load_path is set to 'models/lstm.pt'")
        load_path = "models/lstm.pt"
    
    state_dict = torch.load(
        f=load_path,
        map_location=device
    )
    logger.info(f"Model loaded from <== {load_path}")
    
    model.load_state_dict(state_dict=state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict=state_dict['optimizer'])
    
    return state_dict['valid_loss']



def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    
    if save_path is None:
        os.makedirs(name="metrics", exist_ok=True)
        save_path = "metrics/loss.list"
        logger.warning(f"save_path is None. So save_path is set to '{save_path}'")
    
    state_dict = {
        'train_loss_list': train_loss_list,
        'valid_loss_list': valid_loss_list,
        'global_steps_list': global_steps_list
    }
    
    torch.save(obj=state_dict, f=save_path)
    logger.info(f"Model saved to ==> {save_path}")



def load_metrics(load_path, device):
    if load_path == None:
        load_path = "metrics/loss.list"
        logger.warning(f"load_path is None. So load_path is set to '{load_path}'")
    
    state_dict = torch.load(f=load_path, map_location=device)
    logger.info(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']



def train(
    model:RealFakeDetection,
    optimizer:optim.Optimizer,
    device,
    train_loader,
    valid_loader,
    criterion=nn.BCELoss(),
    num_epochs=5,
    file_path="models",
    best_valid_loss=float("Inf")
):
    eval_every = len(train_loader) // 2
    
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    logger.info("================== Start training ==================")
    model.train()
    for epoch in range(num_epochs):
        for (labels, titletext, titletext_len) in train_loader:
            labels = labels.to(device)
            titletext = titletext.to(device)
            titletext_len = titletext_len
            output = model(
                text=titletext,
                text_len=titletext_len
            )
            
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update running values
            running_loss += loss.item()
            global_step += 1 
            
            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                
                with torch.no_grad():
                    # validation loop
                    for (labels, titletext, titletext_len) in valid_loader:
                        labels = labels.to(device)
                        titletext = titletext.to(device)
                        titletext_len = titletext_len
                        output = model(titletext, titletext_len)
                        
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()
                
                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                
                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()
                
                # print progress
                logger.info(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'\
                        .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader), average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_train_loss
                    save_checkpoint(
                        save_path=file_path + "/model.pt",
                        model=model,
                        optimizer=optimizer,
                        valid_loss=best_valid_loss
                    )
                    save_metrics(
                        valid_loss_list=valid_loss_list,
                        global_steps_list=global_steps_list,
                        save_path=file_path + "/metrics.pt",
                        train_loss_list=train_loss_list
                    )
    save_metrics(
        valid_loss_list=valid_loss_list,
        global_steps_list=global_steps_list,
        save_path=file_path + "/metrics.pt",
        train_loss_list=train_loss_list
    )
    logger.info("Finished Training!")