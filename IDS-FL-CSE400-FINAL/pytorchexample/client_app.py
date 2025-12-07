"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from pytorchexample.task import get_weights, load_data, set_weights, test, train, Net
from pytorchexample.getdist import get_class_distribution
from pytorchexample.communication_utils import (
    calculate_and_log_communication, 
    get_zero_parameters_for_rejected_client,
    comm_tracker
)

# Define Flower Client

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, testloader, multiclass_loader, multiclass_val_loader, pos_weight, attack_class_weights, local_epochs, learning_rate, cid: int):
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Client {cid} using device: {self.device}")
        
        self.net = Net().to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.multiclass_loader = multiclass_loader
        self.multiclass_val_loader = multiclass_val_loader
        self.pos_weight = pos_weight.to(self.device) if pos_weight is not None else None
        self.attack_class_weights = attack_class_weights.to(self.device) if attack_class_weights is not None else None
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.cid = cid
        self.current_round = 0  # Track current round for communication logging
    

    def fit(self, parameters, config):
        # Increment round counter
        self.current_round += 1
        
        set_weights(self.net, parameters)
        temp = float(config.get("temp", 0.0))
        results, client_accept = train(self.net, self.trainloader, self.valloader, self.multiclass_loader, 
                       self.multiclass_val_loader, self.pos_weight, self.attack_class_weights, 
                       self.local_epochs, self.learning_rate, self.device, temp, self.cid)
        results["accept"] = client_accept
        
        # COMMUNICATION OPTIMIZATION: Return different parameters based on SA decision
        if client_accept:
            # Client accepted by SA - send full model parameters
            model_weights = get_weights(self.net)
        else:
            # Client rejected by SA - send zero parameters to save communication
            full_weights = get_weights(self.net)
            model_weights = get_zero_parameters_for_rejected_client(full_weights)
        
        # Log communication event for analysis
        calculate_and_log_communication(
            client_id=self.cid,
            round_num=self.current_round,
            accepted=client_accept,
            parameters=model_weights,
            metrics={
                "val_accuracy": results.get("val_accuracy", 0.0),
                "val_precision": results.get("val_precision", 0.0),
                "val_recall": results.get("val_recall", 0.0),
                "val_f1": results.get("val_f1", 0.0),
                "val_fpr": results.get("val_fpr", 0.0)
            }
        )
        
        return model_weights, len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, test_size, output_dict = test(self.net, self.testloader, self.device, self.cid)
        return float(loss), test_size, output_dict


                                                         

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader, testloader, multiclass_loader, multiclass_val_loader, pos_weight, attack_class_weights = load_data(partition_id, num_partitions, batch_size)
    # get_class_distribution(partition_id, trainloader, "Training data class distribution")
    # get_class_distribution(partition_id, valloader, "Validation data class distribution")
    # get_class_distribution(partition_id, testloader, "Test data class distribution")    
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, testloader, multiclass_loader, multiclass_val_loader, pos_weight, attack_class_weights, local_epochs, learning_rate, partition_id).to_client()


# Flower ClientApp
app = ClientApp(client_fn = client_fn)