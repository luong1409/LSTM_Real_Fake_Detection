from utils import *
from model import RealFakeDetection, train
from torch import optim

vocab = get_vocab_from()
vocab_size = len(vocab)

train_loader = get_dataloader(path="data/train.csv")
valid_loader = get_dataloader(path="data/valid.csv")

model = RealFakeDetection(max_features=vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
print("All done ---> Ready to train")


if __name__ == "__main__":
    train(
        model=model,
        optimizer=optimizer,
        num_epochs=100,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader
    )