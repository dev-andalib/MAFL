"""
Communication Cost Analysis and Optimization for Simulated Annealing Federated Learning
"""

import torch
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import sys
from datetime import datetime

class CommunicationTracker:
    """Track and analyze communication costs in federated learning with simulated annealing."""
    
    def __init__(self, output_folder="communication_metrics2/"):
        self.output_folder = output_folder
        self.ensure_output_folder()
        self.cleanup_previous_results()
        
    def ensure_output_folder(self):
        """Create output folder if it doesn't exist."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
    def cleanup_previous_results(self):
        """Delete existing communication result files before starting new simulation."""
        files_to_clean = [
            "communication_log.json",
            "communication_savings_analysis.json",
            "communication_savings_summary.json",
            "summary_stats.json",
            "analysis_data.json"
        ]
        
        for filename in files_to_clean:
            filepath = os.path.join(self.output_folder, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"🗑️  Cleaned up previous file: {filename}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not delete {filename}: {e}")
        
        # Clean up individual client round files and intermediate reports
        if os.path.exists(self.output_folder):
            for filename in os.listdir(self.output_folder):
                should_delete = (
                    (filename.startswith("client_") and filename.endswith(".json") and "_round_" in filename) or
                    (filename.startswith("round_") and filename.endswith("_stats.json")) or
                    (filename.startswith("intermediate_report_")) or
                    (filename.startswith("error_round_"))
                )
                if should_delete:
                    try:
                        os.remove(os.path.join(self.output_folder, filename))
                    except Exception as e:
                        print(f"⚠️  Warning: Could not delete {filename}: {e}")
        
        # Also clean up analysis folder
        analysis_folder = "communication_analysis/"
        if os.path.exists(analysis_folder):
            for filename in os.listdir(analysis_folder):
                if filename.endswith(('.json', '.png', '.md', '.csv')):
                    try:
                        os.remove(os.path.join(analysis_folder, filename))
                    except Exception as e:
                        print(f"⚠️  Warning: Could not delete analysis file {filename}: {e}")
        
        print(f"✅ Communication tracking initialized - all previous results cleaned up")
    
    def calculate_model_size_bytes(self, parameters: List[np.ndarray]) -> int:
        """Calculate total size of model parameters in bytes."""
        total_bytes = 0
        for param in parameters:
            # Each float32 parameter is 4 bytes
            total_bytes += param.nbytes
        return total_bytes
    
    def calculate_model_size_mb(self, parameters: List[np.ndarray]) -> float:
        """Calculate total size of model parameters in MB."""
        bytes_size = self.calculate_model_size_bytes(parameters)
        return bytes_size / (1024 * 1024)  # Convert to MB
    
    def get_parameter_breakdown(self, net) -> Dict[str, Dict[str, Any]]:
        """Get detailed breakdown of model parameters by layer."""
        breakdown = {}
        total_params = 0
        total_bytes = 0
        
        for name, param in net.named_parameters():
            param_count = param.numel()
            param_bytes = param.nbytes
            param_shape = list(param.shape)
            
            breakdown[name] = {
                "shape": param_shape,
                "param_count": param_count,
                "bytes": param_bytes,
                "mb": param_bytes / (1024 * 1024)
            }
            
            total_params += param_count
            total_bytes += param_bytes
        
        breakdown["total"] = {
            "total_parameters": total_params,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024)
        }
        
        return breakdown
    
    def create_zero_parameters(self, reference_parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Create zero parameters with same shapes as reference."""
        zero_params = []
        for param in reference_parameters:
            zero_param = np.zeros_like(param)
            zero_params.append(zero_param)
        return zero_params
    
    def log_communication_event(self, client_id: int, round_num: int, 
                                accepted: bool, parameters: List[np.ndarray], 
                                metrics: Dict[str, float] = None):
        """Log a communication event for analysis."""
        
        if accepted:
            comm_size_bytes = self.calculate_model_size_bytes(parameters)
            comm_size_mb = self.calculate_model_size_mb(parameters)
            comm_type = "full_model"
        else:
            # For rejected clients, we send zero parameters (minimal communication)
            comm_size_bytes = 0  # Or minimal metadata size
            comm_size_mb = 0.0
            comm_type = "rejected"
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "round": round_num,
            "accepted": accepted,
            "communication_type": comm_type,
            "size_bytes": comm_size_bytes,
            "size_mb": comm_size_mb,
            "metrics": metrics or {}
        }
        
        # Save individual event
        event_file = os.path.join(self.output_folder, f"client_{client_id}_round_{round_num}.json")
        with open(event_file, 'w') as f:
            json.dump(event, f, indent=4)
        
        # Append to global log
        global_log = os.path.join(self.output_folder, "communication_log.json")
        if os.path.exists(global_log):
            with open(global_log, 'r') as f:
                try:
                    events = json.load(f)
                    if not isinstance(events, list):
                        events = [events]
                except json.JSONDecodeError:
                    events = []
        else:
            events = []
        
        events.append(event)
        
        with open(global_log, 'w') as f:
            json.dump(events, f, indent=4)
    
    def calculate_savings_analysis(self) -> Dict[str, Any]:
        """Calculate comprehensive communication savings analysis."""
        global_log = os.path.join(self.output_folder, "communication_log.json")
        
        if not os.path.exists(global_log):
            return {"error": "No communication log found"}
        
        with open(global_log, 'r') as f:
            events = json.load(f)
        
        total_events = len(events)
        accepted_events = [e for e in events if e["accepted"]]
        rejected_events = [e for e in events if not e["accepted"]]
        
        # Calculate actual communication costs
        actual_comm_mb = sum(e["size_mb"] for e in events)
        actual_comm_bytes = sum(e["size_bytes"] for e in events)
        
        # Calculate what communication would be without SA (all clients send full models)
        if accepted_events:
            avg_model_size_mb = np.mean([e["size_mb"] for e in accepted_events])
            avg_model_size_bytes = np.mean([e["size_bytes"] for e in accepted_events])
        else:
            avg_model_size_mb = 0
            avg_model_size_bytes = 0
        
        without_sa_comm_mb = total_events * avg_model_size_mb
        without_sa_comm_bytes = total_events * avg_model_size_bytes
        
        # Calculate savings
        saved_mb = without_sa_comm_mb - actual_comm_mb
        saved_bytes = without_sa_comm_bytes - actual_comm_bytes
        
        if without_sa_comm_mb > 0:
            savings_percentage = (saved_mb / without_sa_comm_mb) * 100
        else:
            savings_percentage = 0
        
        # Time savings (approximate - assuming communication time is proportional to data size)
        # Assuming 10 Mbps network speed as baseline
        network_speed_mbps = 10  # Adjustable parameter
        
        time_without_sa_seconds = (without_sa_comm_mb * 8) / network_speed_mbps  # Convert MB to Mbits
        time_with_sa_seconds = (actual_comm_mb * 8) / network_speed_mbps
        time_saved_seconds = time_without_sa_seconds - time_with_sa_seconds
        
        # Round-wise analysis
        rounds = set(e["round"] for e in events)
        round_analysis = {}
        
        for round_num in rounds:
            round_events = [e for e in events if e["round"] == round_num]
            round_accepted = [e for e in round_events if e["accepted"]]
            round_rejected = [e for e in round_events if not e["accepted"]]
            
            round_analysis[f"round_{round_num}"] = {
                "total_clients": len(round_events),
                "accepted_clients": len(round_accepted),
                "rejected_clients": len(round_rejected),
                "acceptance_rate": len(round_accepted) / len(round_events) if round_events else 0,
                "communication_mb": sum(e["size_mb"] for e in round_events),
                "potential_communication_mb": len(round_events) * avg_model_size_mb,
                "round_savings_mb": (len(round_events) * avg_model_size_mb) - sum(e["size_mb"] for e in round_events)
            }
        
        analysis = {
            "total_communication_events": total_events,
            "accepted_clients": len(accepted_events),
            "rejected_clients": len(rejected_events),
            "acceptance_rate": len(accepted_events) / total_events if total_events > 0 else 0,
            "communication_costs": {
                "actual_mb": actual_comm_mb,
                "actual_bytes": actual_comm_bytes,
                "without_sa_mb": without_sa_comm_mb,
                "without_sa_bytes": without_sa_comm_bytes,
                "average_model_size_mb": avg_model_size_mb,
                "average_model_size_bytes": avg_model_size_bytes
            },
            "savings": {
                "saved_mb": saved_mb,
                "saved_bytes": saved_bytes,
                "savings_percentage": savings_percentage,
                "communication_reduction_factor": without_sa_comm_mb / actual_comm_mb if actual_comm_mb > 0 else 0
            },
            "time_analysis": {
                "assumed_network_speed_mbps": network_speed_mbps,
                "time_without_sa_seconds": time_without_sa_seconds,
                "time_with_sa_seconds": time_with_sa_seconds,
                "time_saved_seconds": time_saved_seconds,
                "time_saved_minutes": time_saved_seconds / 60,
                "time_saved_hours": time_saved_seconds / 3600,
                "time_savings_percentage": (time_saved_seconds / time_without_sa_seconds * 100) if time_without_sa_seconds > 0 else 0
            },
            "round_wise_analysis": round_analysis
        }
        
        # Save analysis
        analysis_file = os.path.join(self.output_folder, "communication_savings_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=4)
        
        return analysis

    def save_savings_summary_json(self, analysis: Dict[str, Any]) -> str:
        """Save a comprehensive summary of communication savings to JSON file."""
        if "error" in analysis:
            error_summary = {"error": analysis["error"], "timestamp": datetime.now().isoformat()}
            error_file = os.path.join(self.output_folder, "communication_error.json")
            with open(error_file, 'w') as f:
                json.dump(error_summary, f, indent=4)
            return error_file
        
        # Create comprehensive summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "simulation_info": {
                "title": "Simulated Annealing Communication Savings Analysis",
                "description": "Comprehensive analysis of communication costs and savings in federated learning with SA client selection"
            },
            "overall_statistics": {
                "total_communication_events": analysis['total_communication_events'],
                "accepted_clients": analysis['accepted_clients'],
                "rejected_clients": analysis['rejected_clients'],
                "acceptance_rate": analysis['acceptance_rate']
            },
            "communication_costs": {
                "average_model_size_mb": analysis['communication_costs']['average_model_size_mb'],
                "actual_communication_mb": analysis['communication_costs']['actual_mb'],
                "without_sa_communication_mb": analysis['communication_costs']['without_sa_mb'],
                "average_model_size_bytes": analysis['communication_costs']['average_model_size_bytes'],
                "actual_communication_bytes": analysis['communication_costs']['actual_bytes'],
                "without_sa_communication_bytes": analysis['communication_costs']['without_sa_bytes']
            },
            "savings": {
                "data_saved_mb": analysis['savings']['saved_mb'],
                "data_saved_bytes": analysis['savings']['saved_bytes'],
                "savings_percentage": analysis['savings']['savings_percentage'],
                "communication_reduction_factor": analysis['savings']['communication_reduction_factor']
            },
            "time_analysis": {
                "network_speed_assumed_mbps": analysis['time_analysis']['assumed_network_speed_mbps'],
                "time_without_sa_seconds": analysis['time_analysis']['time_without_sa_seconds'],
                "time_with_sa_seconds": analysis['time_analysis']['time_with_sa_seconds'],
                "time_saved_seconds": analysis['time_analysis']['time_saved_seconds'],
                "time_saved_minutes": analysis['time_analysis']['time_saved_minutes'],
                "time_saved_hours": analysis['time_analysis']['time_saved_hours'],
                "time_savings_percentage": analysis['time_analysis']['time_savings_percentage']
            },
            "round_wise_summary": []
        }
        
        # Add round-wise summary
        if 'round_wise_analysis' in analysis:
            for round_key, round_data in analysis['round_wise_analysis'].items():
                round_summary = {
                    "round": round_key,
                    "total_clients": round_data['total_clients'],
                    "accepted_clients": round_data['accepted_clients'],
                    "rejected_clients": round_data['rejected_clients'],
                    "acceptance_rate": round_data['acceptance_rate'],
                    "communication_mb": round_data['communication_mb'],
                    "potential_communication_mb": round_data['potential_communication_mb'],
                    "savings_mb": round_data['round_savings_mb']
                }
                summary["round_wise_summary"].append(round_summary)
        
        # Save summary to JSON file
        summary_file = os.path.join(self.output_folder, "communication_savings_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        return summary_file

# Global tracker instance
comm_tracker = CommunicationTracker()

def get_zero_parameters_for_rejected_client(reference_parameters: List[np.ndarray]) -> List[np.ndarray]:
    """Return zero parameters when client is rejected by SA."""
    return comm_tracker.create_zero_parameters(reference_parameters)

def calculate_and_log_communication(client_id: int, round_num: int, accepted: bool, 
                                  parameters: List[np.ndarray], metrics: Dict[str, float] = None):
    """Calculate and log communication for this client."""
    comm_tracker.log_communication_event(client_id, round_num, accepted, parameters, metrics)

def generate_communication_report() -> Dict[str, Any]:
    """Generate comprehensive communication savings report and save to JSON."""
    analysis = comm_tracker.calculate_savings_analysis()
    summary_file = comm_tracker.save_savings_summary_json(analysis)
    
    # Print minimal confirmation instead of full report
    if "error" not in analysis:
        print(f"📊 Communication analysis complete: {analysis['savings']['savings_percentage']:.1f}% savings achieved")
        print(f"💾 Results saved to: {summary_file}")
    else:
        print(f"❌ Communication analysis error - check: {summary_file}")
    
    return analysis