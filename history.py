from matplotlib import pyplot as plt


class History:
    def __init__(self):
        self.hyperparams = {}
        self.train_acc_history = []
        self.test_acc_history = []
        self.loss_history = []

    def plot_training_history(self):
        epochs = list(range(1, len(self.train_acc_history) + 1))
        # Create a single figure with two subplots (2 rows, 1 column)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].plot(epochs, self.train_acc_history, label='Train Accuracy')
        axes[0].plot(epochs, self.test_acc_history, label='Test Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(epochs, self.loss_history, label='Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss History')
        axes[1].legend()
        axes[1].grid()

        num_of_hidden_layers = self.hyperparams['num_of_hidden_layers']
        hid_layers_sizes = self.hyperparams['hid_layers_sizes']
        learn_rate = self.hyperparams['learn_rate']
        batch_size = self.hyperparams['batch_size']

        fig.suptitle(
            f'hidden layers = {num_of_hidden_layers}, hidden layers sizes = {hid_layers_sizes}, learning rate = {learn_rate}, batch size = {batch_size}')
        plt.tight_layout()
        plt.savefig(f'history/history-{num_of_hidden_layers}-{hid_layers_sizes}-{learn_rate}-{batch_size}.jpg')
