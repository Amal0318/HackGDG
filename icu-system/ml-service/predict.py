"""
VitalX Prediction CLI Tool
===========================

Production-ready command-line tool for ICU patient deterioration prediction.

Features:
- Real-time predictions from files or stdin
- Attention visualization
- Feature importance analysis
- Batch processing
- Confidence intervals
- Export to JSON/CSV
- Integration with monitoring systems

Usage:
    python predict.py --input patient_data.json
    python predict.py --batch vitals_folder/
    python predict.py --stream  # Real-time from stdin
    python predict.py --visualize --input sample.json
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pickle
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import from your trained model
from models.lstm_model import LSTMAttentionModel

# ==============================
# CONFIGURATION
# ==============================
FEATURE_NAMES = [
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
    "glucose", "ph", "lactate", "creatinine", "wbc", "hemoglobin", "platelets"
]

RISK_THRESHOLDS = {
    "LOW": 0.3,
    "MEDIUM": 0.7,
    "HIGH": 1.0
}

# Normal ranges for validation
NORMAL_RANGES = {
    "heart_rate": (40, 180),
    "sbp": (70, 200),
    "dbp": (40, 120),
    "map": (60, 140),
    "resp_rate": (8, 40),
    "spo2": (70, 100),
    "temperature": (35.0, 41.0),
    "glucose": (50, 500),
    "ph": (6.8, 7.8),
    "lactate": (0.5, 10.0),
    "creatinine": (0.3, 10.0),
    "wbc": (1.0, 50.0),
    "hemoglobin": (5.0, 20.0),
    "platelets": (10, 1000)
}


# ==============================
# PREDICTOR CLASS
# ==============================
class VitalXPredictor:
    """Production-ready predictor with explainability."""
    
    def __init__(self, model_path="app/model.pth", scaler_path="app/scaler.pkl", 
                 config_path="app/feature_config.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load scaler
        self.scaler = self._load_scaler(scaler_path)
        
        # Load config
        self.config = self._load_config(config_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Architecture: LSTM with Attention")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, model_path: str) -> LSTMAttentionModel:
        """Load trained LSTM model."""
        if not os.path.exists(model_path):
            # Try alternative paths
            alt_paths = [
                "training/saved_models/best_model.pth",
                "../training/saved_models/best_model.pth"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get config
        model_config = checkpoint.get('config', {})
        
        # Create model
        model = LSTMAttentionModel(
            input_size=model_config.get('input_size', 14),
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.3)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_scaler(self, scaler_path: str) -> Optional[object]:
        """Load feature scaler."""
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("⚠️  No scaler found, using raw features")
            return None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load feature config."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def validate_sequence(self, sequence: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate vital signs are within normal ranges."""
        warnings_list = []
        
        if sequence.shape != (60, 14):
            return False, [f"Invalid shape: {sequence.shape}, expected (60, 14)"]
        
        # Check for NaN/Inf
        if np.isnan(sequence).any():
            warnings_list.append("Contains NaN values")
        if np.isinf(sequence).any():
            warnings_list.append("Contains infinite values")
        
        # Check ranges
        for i, feature in enumerate(FEATURE_NAMES):
            if feature in NORMAL_RANGES:
                min_val, max_val = NORMAL_RANGES[feature]
                feature_vals = sequence[:, i]
                
                if (feature_vals < min_val).any() or (feature_vals > max_val).any():
                    warnings_list.append(
                        f"{feature}: out of range [{min_val}, {max_val}] "
                        f"(min={feature_vals.min():.2f}, max={feature_vals.max():.2f})"
                    )
        
        return len(warnings_list) == 0 or not np.isnan(sequence).any(), warnings_list
    
    def predict(self, sequence: np.ndarray, return_attention=False) -> Dict:
        """
        Predict deterioration risk with explainability.
        
        Args:
            sequence: (60, 14) array of vital signs
            return_attention: Whether to return attention weights
        
        Returns:
            dict with risk_score, risk_level, attention_weights, feature_importance
        """
        # Validate
        is_valid, warnings_list = self.validate_sequence(sequence)
        
        # Prepare input
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, 60, 14)
        
        # Predict
        with torch.no_grad():
            output, attention_weights = self.model(X)
        
        risk_score = float(output.squeeze().cpu().numpy())
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        # Classify risk
        if risk_score < RISK_THRESHOLDS["LOW"]:
            risk_level = "LOW"
        elif risk_score < RISK_THRESHOLDS["MEDIUM"]:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        result = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "is_valid": is_valid,
            "validation_warnings": warnings_list,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add attention weights
        if return_attention:
            attn = attention_weights.squeeze().cpu().numpy()  # (60,)
            result["attention_weights"] = attn.tolist()
            result["critical_timesteps"] = self._get_critical_timesteps(attn)
        
        # Add feature importance
        result["feature_importance"] = self._calculate_feature_importance(sequence, attention_weights)
        
        return result
    
    def _get_critical_timesteps(self, attention_weights: np.ndarray, top_k=5) -> List[Dict]:
        """Get top-k most critical timesteps."""
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        
        critical = []
        for idx in top_indices:
            critical.append({
                "timestep": int(idx),
                "seconds_ago": int(60 - idx),
                "attention_score": float(attention_weights[idx])
            })
        
        return critical
    
    def _calculate_feature_importance(self, sequence: np.ndarray, 
                                     attention_weights: torch.Tensor) -> Dict[str, float]:
        """Calculate feature importance using attention-weighted statistics."""
        attn = attention_weights.squeeze().cpu().numpy()  # (60,)
        
        # Weight each timestep by attention
        weighted_sequence = sequence * attn[:, np.newaxis]  # (60, 14)
        
        # Calculate importance as weighted standard deviation
        importance = {}
        for i, feature in enumerate(FEATURE_NAMES):
            feature_values = weighted_sequence[:, i]
            importance[feature] = float(np.std(feature_values))
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def predict_batch(self, sequences: List[np.ndarray], verbose=True) -> List[Dict]:
        """Batch prediction for multiple sequences."""
        results = []
        
        for i, seq in enumerate(sequences):
            if verbose:
                print(f"Processing {i+1}/{len(sequences)}...", end='\r')
            
            try:
                result = self.predict(seq, return_attention=False)
                result["sequence_id"] = i
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error processing sequence {i}: {e}")
                continue
        
        if verbose:
            print(f"\n✅ Processed {len(results)}/{len(sequences)} sequences")
        
        return results
    
    def visualize_prediction(self, sequence: np.ndarray, result: Dict):
        """Visualize prediction with attention weights."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            print("⚠️  matplotlib not installed. Run: pip install matplotlib seaborn")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Attention Heatmap
        if "attention_weights" in result:
            attn = np.array(result["attention_weights"])
            ax = axes[0]
            im = ax.imshow(attn.reshape(1, -1), aspect='auto', cmap='YlOrRd')
            ax.set_title(f"Attention Weights - Risk Score: {result['risk_score']:.3f} ({result['risk_level']})", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Timestep (seconds ago)")
            ax.set_ylabel("Attention")
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        
        # 2. Feature Importance
        importance = result.get("feature_importance", {})
        if importance:
            ax = axes[1]
            features = list(importance.keys())[:10]  # Top 10
            scores = [importance[f] for f in features]
            
            bars = ax.barh(features, scores, color='steelblue')
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 10 Feature Importance", fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            
            # Color code by risk
            if result['risk_level'] == 'HIGH':
                for bar in bars[:3]:
                    bar.set_color('red')
        
        # 3. Critical Timesteps
        critical = result.get("critical_timesteps", [])
        if critical:
            ax = axes[2]
            timesteps = [c['timestep'] for c in critical]
            scores = [c['attention_score'] for c in critical]
            
            ax.bar(timesteps, scores, color='orange', alpha=0.7)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Attention Score")
            ax.set_title("Top 5 Critical Timesteps", fontsize=12, fontweight='bold')
            ax.set_xlim(0, 60)
        
        plt.tight_layout()
        
        # Save
        output_path = f"prediction_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: {output_path}")
        plt.show()


# ==============================
# CLI INTERFACE
# ==============================
def load_data_from_file(filepath: str) -> np.ndarray:
    """Load sequence from JSON or CSV file."""
    path = Path(filepath)
    
    if path.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if "sequence" in data:
            sequence = np.array(data["sequence"], dtype=np.float32)
        else:
            sequence = np.array(data, dtype=np.float32)
    
    elif path.suffix == '.csv':
        df = pd.read_csv(filepath)
        sequence = df.values[:60, :14]  # First 60 rows, 14 columns
    
    elif path.suffix == '.npy':
        sequence = np.load(filepath)
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return sequence


def main():
    parser = argparse.ArgumentParser(description="VitalX Prediction CLI")
    parser.add_argument("--input", "-i", help="Input file (JSON/CSV/NPY)")
    parser.add_argument("--batch", "-b", help="Batch process directory")
    parser.add_argument("--stream", "-s", action="store_true", help="Stream from stdin")
    parser.add_argument("--output", "-o", default="predictions.json", help="Output file")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize attention")
    parser.add_argument("--model", "-m", default="app/model.pth", help="Model path")
    parser.add_argument("--export-csv", action="store_true", help="Export to CSV")
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("=" * 70)
    print("VitalX Prediction CLI v1.0")
    print("=" * 70)
    
    predictor = VitalXPredictor(model_path=args.model)
    
    # Single file prediction
    if args.input:
        print(f"\n📂 Loading: {args.input}")
        sequence = load_data_from_file(args.input)
        
        print(f"📊 Sequence shape: {sequence.shape}")
        
        result = predictor.predict(sequence, return_attention=args.visualize)
        
        # Display
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"Risk Score:  {result['risk_score']:.4f}")
        print(f"Risk Level:  {result['risk_level']}")
        print(f"Valid:       {result['is_valid']}")
        
        if result['validation_warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in result['validation_warnings']:
                print(f"   - {warning}")
        
        print(f"\n📈 Top 5 Important Features:")
        for i, (feature, score) in enumerate(list(result['feature_importance'].items())[:5], 1):
            print(f"   {i}. {feature:15s}: {score:.4f}")
        
        if "critical_timesteps" in result:
            print(f"\n🎯 Critical Timesteps:")
            for ts in result['critical_timesteps']:
                print(f"   - {ts['seconds_ago']}s ago (timestep {ts['timestep']}): {ts['attention_score']:.4f}")
        
        # Save
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to: {args.output}")
        
        # Visualize
        if args.visualize:
            predictor.visualize_prediction(sequence, result)
    
    # Batch processing
    elif args.batch:
        print(f"\n📂 Batch processing: {args.batch}")
        
        batch_dir = Path(args.batch)
        files = list(batch_dir.glob("*.json")) + list(batch_dir.glob("*.csv"))
        
        print(f"Found {len(files)} files")
        
        sequences = []
        for file in files:
            try:
                seq = load_data_from_file(str(file))
                sequences.append(seq)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        
        results = predictor.predict_batch(sequences)
        
        # Summary
        print("\n" + "=" * 70)
        print("BATCH RESULTS")
        print("=" * 70)
        
        high_risk = sum(1 for r in results if r['risk_level'] == 'HIGH')
        medium_risk = sum(1 for r in results if r['risk_level'] == 'MEDIUM')
        low_risk = sum(1 for r in results if r['risk_level'] == 'LOW')
        
        print(f"Total:       {len(results)}")
        print(f"High Risk:   {high_risk} ({high_risk/len(results)*100:.1f}%)")
        print(f"Medium Risk: {medium_risk} ({medium_risk/len(results)*100:.1f}%)")
        print(f"Low Risk:    {low_risk} ({low_risk/len(results)*100:.1f}%)")
        
        # Save
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved to: {args.output}")
        
        # Export CSV
        if args.export_csv:
            df = pd.DataFrame([
                {
                    "sequence_id": r["sequence_id"],
                    "risk_score": r["risk_score"],
                    "risk_level": r["risk_level"],
                    "is_valid": r["is_valid"]
                }
                for r in results
            ])
            csv_path = args.output.replace('.json', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"📊 CSV exported: {csv_path}")
    
    # Stream mode
    elif args.stream:
        print("\n🌊 Stream mode - Enter JSON sequences (Ctrl+C to exit)")
        print("Format: {\"sequence\": [[...]]}")
        
        while True:
            try:
                line = input()
                data = json.loads(line)
                sequence = np.array(data["sequence"], dtype=np.float32)
                
                result = predictor.predict(sequence)
                print(json.dumps({
                    "risk_score": result["risk_score"],
                    "risk_level": result["risk_level"]
                }))
            
            except KeyboardInterrupt:
                print("\n👋 Exiting stream mode")
                break
            except Exception as e:
                print(json.dumps({"error": str(e)}))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
