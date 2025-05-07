import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd

def plot_model_performance(json_file, output_dir=None, num_bins=500):
    """
    Load model performance data from a JSON file and plot it with total fidelity.
    
    Args:
        json_file (str): Path to the JSON file containing model performance data
        output_dir (str, optional): Directory to save the output image. 
                                   If None, uses the same directory as the JSON file.
        num_bins (int, optional): Number of points to display per curve. Default is 500.
    """
    # Load the data from JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract the x and y values from the first element in the array
    x_values = data[0]['x']
    y_values = data[0]['y']
    model_name = data[0].get('name', 'Model Performance')
    
    # Create a pandas DataFrame for easier binning
    df = pd.DataFrame({
        'step': x_values,
        'value': y_values
    })
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set axis labels and title
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    filename_no_ext = os.path.splitext(os.path.basename(json_file))[0]
    ax.set_title(f'{filename_no_ext}', fontsize=14)
    
    # Total number of points
    total_points = len(df)
    
    # If we have more points than bins, perform binning
    if total_points > num_bins:
        # Add bin column to DataFrame
        df['bin'] = pd.cut(df.index, bins=num_bins, labels=False)
        
        # Get min and max values for each bin
        bin_mins = df.groupby('bin').min()
        bin_maxs = df.groupby('bin').max()
        
        # Plot the max values (upper envelope)
        ax.plot(bin_maxs['step'], bin_maxs['value'], color='#E74C3C', linewidth=1, label='Max')
        
        # Plot the min values (lower envelope) and fill between
        ax.plot(bin_mins['step'], bin_mins['value'], color='#E74C3C', linewidth=1, label='Min')
        
        # Fill between min and max
        ax.fill_between(bin_mins['step'], bin_mins['value'], bin_maxs['value'], color='#E74C3C', alpha=0.3)
    else:
        # If we have fewer points than bins, just plot all points
        ax.plot(df['step'], df['value'], color='#E74C3C', linewidth=1)
    
    # Update title with fidelity information
    ax.set_title(f"Total Fidelity: {filename_no_ext}", fontsize=14)
    
    # Format the x-axis to display in thousands (k)
    def format_x_tick(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}k'
        return str(int(x))
    
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(format_x_tick))
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # custom_yticks = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    # ax.set_yticks(custom_yticks)
    # ax.set_yticklabels([str(y) for y in custom_yticks])
    
    ax.set_ylim([0.0005, 0.002])
    
    # Add grid lines
    ax.grid(True, linestyle='-', alpha=0.2, which='both')
    
    # Set the background color to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Add a thin border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#E0E0E0')
        spine.set_linewidth(1)
    
    # Tight layout
    plt.tight_layout()
    
    # Determine output file path
    if output_dir is None:
        output_dir = os.path.dirname(json_file)
    
    output_file = os.path.join(output_dir, f"{filename_no_ext}.png")
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Close the figure to free memory
    plt.close(fig)

def process_folder(input_dir, output_dir=None, num_bins=500):
    """
    Process all JSON files in the input directory and generate plots.
    
    Args:
        input_dir (str): Path to the directory containing JSON files
        output_dir (str, optional): Directory to save the output images.
                                   If None, uses the same directory as the JSON files.
        num_bins (int, optional): Number of points to display per curve. Default is 500.
    """
    # If output_dir is not specified, use input_dir
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        try:
            plot_model_performance(json_path, output_dir, num_bins)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate model performance graphs from JSON data')
    parser.add_argument('input', help='Path to JSON file or directory containing JSON files')
    parser.add_argument('--output', '-o', help='Directory to save the output images (optional)')
    parser.add_argument('--bins', '-b', type=int, default=500, help='Number of bins for the visualization (default: 500)')
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single file
        if args.input.endswith('.json'):
            output_dir = args.output if args.output else os.path.dirname(args.input)
            plot_model_performance(args.input, output_dir, args.bins)
        else:
            print(f"Input file must be a JSON file: {args.input}")
    elif os.path.isdir(args.input):
        # Process all JSON files in directory
        process_folder(args.input, args.output, args.bins)
    else:
        print(f"Input path does not exist: {args.input}")