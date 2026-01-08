import os
import re
import pandas as pd
import glob
import numpy as np

LOG_DIR = "paper_results/logs"
OUTPUT_FILE = "paper_results/recovered_cnn_cfc_summary.csv"

def parse_log_file(filepath):
    filename = os.path.basename(filepath)
    
    parts = filename.replace('.log', '').split('_')
    
    # Dataset / Backbone Detection
    if filename.startswith("cifar"):
        dataset = "seq-cifar10"
        backbone = "cnn-cfc"
    elif filename.startswith("mnist"):
        dataset = "seq-mnist"
        backbone = "mnistcfc"
    elif filename.startswith("tep"):
        dataset = "tep-anomaly"
        backbone = "tepcfc"
    else:
        dataset = "unknown"
        backbone = "unknown"
    
    seed = parts[-1].replace('seed', '')
    
    # Model / Buffer Detection
    if len(parts) >= 3:
        model_part = parts[2]
        buffer_match = re.search(r'(\D+)(\d+)', model_part)
        if buffer_match:
            model = buffer_match.group(1)
            buffer_size = int(buffer_match.group(2))
        else:
            model = model_part
            buffer_size = 0
    else:
        model = "unknown"
        buffer_size = 0
        
    final_class_il_mean = None
    final_task_il_mean = None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # regex for mean accuracy: "RESULT_class_mean_accs 19.7"
            match_mean = re.search(r"RESULT_class_mean_accs\s+([\d\.]+)", content)
            if match_mean:
                final_class_il_mean = float(match_mean.group(1))
            
            # fallback: calculate from list if explicit mean not found
            # "Raw accuracy values: Class-IL [0.0, 0.0, ...]"
            if final_class_il_mean is None:
                match_list = re.search(r"Raw accuracy values: Class-IL \[(.*?)\]", content)
                if match_list:
                    vals = [float(x) for x in match_list.group(1).split(',')]
                    if vals:
                        final_class_il_mean = sum(vals) / len(vals)

            # Task-IL mean
            match_task_mean = re.search(r"RESULT_task_mean_accs\s+([\d\.]+)", content) # less common?
            if match_task_mean:
                final_task_il_mean = float(match_task_mean.group(1))
            else:
                 # fallback to list
                match_task_list = re.search(r"Raw accuracy values: .*? Task-IL \[(.*?)\]", content)
                if match_task_list:
                    vals = [float(x) for x in match_task_list.group(1).split(',')]
                    if vals:
                        final_task_il_mean = sum(vals) / len(vals)

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    return {
        "filename": filename,
        "dataset": dataset,
        "backbone": backbone,
        "model": model,
        "buffer_size": buffer_size,
        "seed": seed,
        "final_class_il_mean": final_class_il_mean,
        "final_task_il_mean": final_task_il_mean
    }

def main():
    log_files = glob.glob(os.path.join(LOG_DIR, "*cfc*.log"))
    results = []
    
    print(f"Found {len(log_files)} logs matching 'cfc'. Processing...")
    
    for log_file in log_files:
        res = parse_log_file(log_file)
        # Filter for CIFAR results specifically (CNN-CfC)
        if res and res['dataset'] == 'seq-cifar10' and res['final_class_il_mean'] is not None:
            results.append(res)
            
    df = pd.DataFrame(results)
    if not df.empty:
        df['final_class_il_mean'] = pd.to_numeric(df['final_class_il_mean'])
        df.sort_values(by=['model', 'buffer_size', 'seed'], inplace=True)
        
        print("\n=== RECOVERED CNN-CFC RESULTS (CIFAR-10) ===")
        print(df[['model', 'buffer_size', 'seed', 'final_class_il_mean']].to_string())
        
        summary = df.groupby(['model', 'buffer_size'])['final_class_il_mean'].agg(['mean', 'std', 'count'])
        print("\n=== AGGREGATED STATISTICS ===")
        print(summary)
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved detailed summary to {OUTPUT_FILE}")
    else:
        print("No valid seq-cifar10 results extracted.")

if __name__ == "__main__":
    main()
