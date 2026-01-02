"""pytorchexample: A Flower / PyTorch app."""
import flwr
from typing import List, Tuple, Dict, Union
from flwr.common import Context,  ndarrays_to_parameters, FitRes, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from Federated_Learning.task import get_weights, Net, test, set_weights
import numpy as np
import torch
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader
from Federated_Learning.result_visualizer import plot_and_save_averaged_metrics, plot_and_save_energy_temp
from Federated_Learning.communication_utils import generate_communication_report, comm_tracker
import time




class SA(FedAvg): 
    """Custom FedAvg strategy that handles models with different multiclass head sizes and tracks communication."""
    def __init__(self, start_temp = 0.02, cooling=0.99, **kwargs):
        super().__init__(**kwargs)
        self.start_temp = start_temp
        self.cooling = cooling
        self.current_round = 0
        
    def configure_fit(self, server_round: int, parameters, client_manager):
        self.current_round = server_round
        fit_ins = super().configure_fit(server_round, parameters, client_manager)    
        for _, fin in fit_ins:
            cfg = dict(fin.config or {})
            cfg['temp'] = float(self.start_temp)
            fin.config = cfg
        self.start_temp *= self.cooling
        return fit_ins
    

    def aggregate_fit(self, server_round: int, results: List[Tuple[flwr.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[flwr.server.client_proxy.ClientProxy, FitRes], BaseException]],) -> Tuple[flwr.common.Parameters, dict]:
        
        # Track communication statistics for this round
        total_clients = len(results)
        accepted_clients = 0
        rejected_clients = 0
        
        # Filter accepted clients (those that sent non-zero parameters)
        accepted_results = []
        for client_proxy, fit_res in results:
            client_accepted = fit_res.metrics.get("accept", False)
            
            if client_accepted:
                # Check if parameters are non-zero (actual model updates)
                parameters = parameters_to_ndarrays(fit_res.parameters)
                is_zero_params = all(np.allclose(param, 0.0) for param in parameters)
                
                if not is_zero_params:
                    accepted_results.append((client_proxy, fit_res))
                    accepted_clients += 1
                else:
                    rejected_clients += 1
            else:
                rejected_clients += 1
        
        # Save round statistics to JSON
        round_stats = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "total_clients": total_clients,
            "accepted_clients": accepted_clients,
            "rejected_clients": rejected_clients,
            "acceptance_rate": accepted_clients/total_clients if total_clients > 0 else 0
        }
        
        # Calculate communication savings for this round
        if accepted_results:
            sample_params = parameters_to_ndarrays(accepted_results[0][1].parameters)
            model_size_mb = comm_tracker.calculate_model_size_mb(sample_params)
            saved_mb = rejected_clients * model_size_mb
            round_stats["model_size_mb"] = model_size_mb
            round_stats["communication_saved_mb"] = saved_mb
        
        # Save round statistics
        round_stats_file = os.path.join(comm_tracker.output_folder, f"round_{server_round}_stats.json")
        with open(round_stats_file, 'w') as f:
            json.dump(round_stats, f, indent=4)
        
        # Minimal console output
        print(f"Round {server_round}: {accepted_clients}/{total_clients} clients accepted ({accepted_clients/total_clients:.1%})")
        
        # If no clients accepted, return old parameters
        if not accepted_results:
            return None, {}

        # Call original FedAvg aggregation on accepted results only
        return super().aggregate_fit(server_round, accepted_results, failures)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Calculate weighted average of metrics across clients, including recall and precision."""
    total_examples = sum([num for num, _ in metrics])

    # Weighted averages for accuracy
    weighted_binary_acc = sum([metric.get("binary_acc", 0) * num for num, metric in metrics])
    weighted_multi_acc = sum([metric.get("multi_acc", 0) * num for num, metric in metrics])
    
    # Weighted averages for recall and precision
    weighted_binary_recall = sum([metric.get("binary_recall", 0) * num for num, metric in metrics])
    weighted_binary_precision = sum([metric.get("binary_precision", 0) * num for num, metric in metrics])
    weighted_multi_recall = sum([metric.get("multi_recall", 0) * num for num, metric in metrics])
    weighted_multi_precision = sum([metric.get("multi_precision", 0) * num for num, metric in metrics])

    # Calculate weighted averages for all metrics
    accuracy_avg_binary = weighted_binary_acc / total_examples if total_examples > 0 else 0
    accuracy_avg_multi = weighted_multi_acc / total_examples if total_examples > 0 else 0
    recall_avg_binary = weighted_binary_recall / total_examples if total_examples > 0 else 0
    precision_avg_binary = weighted_binary_precision / total_examples if total_examples > 0 else 0
    recall_avg_multi = weighted_multi_recall / total_examples if total_examples > 0 else 0
    precision_avg_multi = weighted_multi_precision / total_examples if total_examples > 0 else 0

    return {
        "accuracy_avg_binary": accuracy_avg_binary,
        "precision_avg_binary": precision_avg_binary,
        "recall_avg_binary": recall_avg_binary,        
        "accuracy_avg_multi": accuracy_avg_multi,
        "precision_avg_multi": precision_avg_multi,
        "recall_avg_multi": recall_avg_multi,
    }

def fit_weighted_avg(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Calculate weighted average of training/validation metrics across clients, summing confusion matrix elements."""
    total_examples = sum([num for num, _ in metrics])

    weighted_val_losses = sum([metric["val_loss"] * num for num, metric in metrics])
    weighted_accuracies = sum([metric["val_accuracy"] * num for num, metric in metrics])
    weighted_recalls = sum([metric["val_recall"] * num for num, metric in metrics])
    weighted_precisions = sum([metric["val_precision"] * num for num, metric in metrics])

    # Calculate weighted averages for rates
    val_loss_avg = weighted_val_losses / total_examples
    accuracy_avg = weighted_accuracies / total_examples
    recall_avg = weighted_recalls / total_examples
    precision_avg = weighted_precisions / total_examples


    return {
        "val_loss": val_loss_avg,
        "val_accuracy": accuracy_avg,
        "val_precision": precision_avg,
        "val_recall": recall_avg,
    }


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = Net(input_features=20, num_attack_types=9)
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, test_size, output_dict = test(net, testloader, device=device, num_classes=9) # dummy temp and prev acc send for now
        return loss, output_dict

    return evaluate


class CommunicationAwareSA(SA):
    """Extended SA strategy with communication reporting."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def aggregate_fit(self, server_round: int, results: List[Tuple[flwr.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[flwr.server.client_proxy.ClientProxy, FitRes], BaseException]],) -> Tuple[flwr.common.Parameters, dict]:
        
        # Call parent aggregation
        result = super().aggregate_fit(server_round, results, failures)
        
        # Generate communication report every 5 rounds or at the end
        if server_round % 5 == 0 or server_round >= 10:  # Adjust based on your total rounds
            try:
                analysis = generate_communication_report()
                
                # Save intermediate report for this round
                intermediate_report = {
                    "round": server_round,
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis
                }
                
                report_file = os.path.join(comm_tracker.output_folder, f"intermediate_report_round_{server_round}.json")
                with open(report_file, 'w') as f:
                    json.dump(intermediate_report, f, indent=4)
                    
            except Exception as e:
                error_report = {
                    "round": server_round,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                error_file = os.path.join(comm_tracker.output_folder, f"error_round_{server_round}.json")
                with open(error_file, 'w') as f:
                    json.dump(error_report, f, indent=4)

        
        if server_round == 2: # Adjust based on your total rounds
            print("Training complete. Generating final metrics plots...")
            try:
                METRICS_BASE = r"D:\T24\MAFL\IDS-FL-CSE400-FINAL_Manuscript Phase\client_metrics" 
                RESULTS_BASE = r"D:\T24\MAFL\IDS-FL-CSE400-FINAL_Manuscript Phase\results"
                plot_and_save_averaged_metrics('train_metrics',METRICS_BASE, RESULTS_BASE)
                plot_and_save_averaged_metrics('val_metrics',METRICS_BASE, RESULTS_BASE)
                plot_and_save_averaged_metrics('test_metrics',METRICS_BASE, RESULTS_BASE)
                plot_and_save_energy_temp(r"D:\T24\MAFL\IDS-FL-CSE400-FINAL_Manuscript Phase\client_sa_metrics" , r"D:\T24\MAFL\IDS-FL-CSE400-FINAL_Manuscript Phase\results")
                
                


            except Exception as e:
                print(f"Failed to generate plots: {e}")

        
        return result

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    

    print(f" Starting FL with SA client selection ({num_rounds} rounds, {context.run_config['min_available_clients']} min clients)")
    print(f"Communication cost tracking enabled - results will be saved to JSON files")

    ndarrays = get_weights(Net(input_features=20, seq_length=10, num_attack_types=9))
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Log model size information
    model_size_mb = comm_tracker.calculate_model_size_mb(ndarrays)
    model_info = {
        "timestamp": datetime.now().isoformat(),
        "model_architecture": "CNN-BiLSTM with Attention",
        "input_features": 20,
        "sequence_length": 10,
        "attack_types": 9,
        "model_size_mb": model_size_mb,
        "total_parameters": sum(param.size for param in ndarrays)
    }
    
    model_info_file = os.path.join(comm_tracker.output_folder, "model_info.json")
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f, indent=4)
    

    # Define the strategy with communication awareness
    strategy = CommunicationAwareSA(
        min_available_clients=context.run_config["min_available_clients"],
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        fit_metrics_aggregation_fn=fit_weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_average,
        # evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)