import argparse
import sys
import matplotlib.pyplot as plt

import torch
import click

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    input, labels = torch.tensor(train_set[0]), torch.tensor(train_set[1])
    all_loss = []
    for epoch in range(100):
        running_loss = 0
        for i in range(len(input)//50):
            images = input[50 * i: 50 * (i + 1)]
            labs = labels[50 * i: 50 * (i + 1)]
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # TODO: Training pass
            output = model.forward(images)
            loss = model.criterion(output, labs.to(torch.long))
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / i}")
            all_loss.append(running_loss)
    plt.plot(all_loss)
    plt.show()
    torch.save(model, "trained_model.pt")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    input, labels = torch.tensor(test_set[0]), torch.tensor(test_set[1])
    images = input.view(input.shape[0], -1)
    output = torch.argmax(model.forward(images), dim=1)
    print("accuracy is " + str((5000 - torch.count_nonzero(output - labels).item())/(5000) * 100) + " %")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    