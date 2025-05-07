import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_model_performance(json_file, output_dir=None):
    """
    Load model performance data from a JSON file and plot it.
    
    Args:
        json_file (str): Path to the JSON file containing model performance data
        output_dir (str, optional): Directory to save the output image. 
                                   If None, uses the same directory as the JSON file.
    """
    # Load the data from JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract the x and y values from the first element in the array
    x_values = data[0]['x']
    y_values = data[0]['y']
    model_name = data[0].get('name', 'Model Performance')
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    ax.plot(x_values, y_values, color='#E74C3C', linewidth=2, marker='o',
            markersize=1, markerfacecolor='#E74C3C', markeredgecolor='#E74C3C')
    
    # Set axis labels and title
    # ax.set_xlabel('Training Steps', fontsize=12)
    filename_no_ext = os.path.splitext(os.path.basename(json_file))[0]
    ax.set_title(f'{filename_no_ext} vs step', fontsize=14)
    
    # Format the x-axis to display in thousands (k)
    def format_x_tick(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}k'
        return str(int(x))
    
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(format_x_tick))
    
    # Determine the magnitude of y values
    max_y_abs = max([abs(y) for y in y_values])
    min_y_abs = min([abs(y) for y in y_values if abs(y) > 0])  # Smallest non-zero value
    
    # Adjust y-axis label and formatting based on the data range
    if max_y_abs <= 1:
        if min_y_abs < 0.001:  # Very small values
            # ax.set_ylabel('Accuracy (Scientific Notation)', fontsize=12)
            # Use scientific notation for very small values
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            # ax.set_ylabel('Accuracy', fontsize=12)
            # Format with appropriate decimal places
            if min_y_abs < 0.01:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.6f}'))
            elif min_y_abs < 0.1:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.4f}'))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    else:
        # ax.set_ylabel('Accuracy', fontsize=12)
        # Use regular formatting for larger values
        ax.yaxis.set_major_formatter(ScalarFormatter())
    
    # Set appropriate y-axis limits with padding
    if all(y == 0 for y in y_values):
        # Handle the case where all values are zero
        ax.set_ylim([-0.1, 0.1])
    else:
        # Calculate appropriate limits based on data
        y_min = min(y_values)
        y_max = max(y_values)
        
        # Add padding, but be careful with very small values
        if y_min >= 0:
            # For strictly positive values
            if min_y_abs < 0.001:  # Very small positive values
                y_lower = 0
                y_upper = y_max * 1.1
            else:
                y_lower = max(0, y_min * 0.9)
                y_upper = y_max * 1.1
        else:
            # For data with negative values
            y_lower = y_min * 1.1
            y_upper = max(0, y_max * 1.1)
        
        ax.set_ylim([y_lower, y_upper])
    
    # # Add grid lines
    # ax.set_yscale('log')
    
    # # Define custom y-axis tick values
    # custom_yticks = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5]
    # ax.set_yticks(custom_yticks)
    # ax.set_yticklabels([str(y) for y in custom_yticks])
    
    # # Set y-axis limits
    # ax.set_ylim([0.002, 5])
    ax.set_ylim([0.00002, 0.6])
    ax.grid(True, linestyle='-', alpha=0.2)
    
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

def process_folder(input_dir, output_dir=None):
    """
    Process all JSON files in the input directory and generate plots.
    
    Args:
        input_dir (str): Path to the directory containing JSON files
        output_dir (str, optional): Directory to save the output images.
                                   If None, uses the same directory as the JSON files.
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
            plot_model_performance(json_path, output_dir)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate model performance graphs from JSON data')
    parser.add_argument('input', help='Path to JSON file or directory containing JSON files')
    parser.add_argument('--output', '-o', help='Directory to save the output images (optional)')
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single file
        if args.input.endswith('.json'):
            output_dir = args.output if args.output else os.path.dirname(args.input)
            plot_model_performance(args.input, output_dir)
        else:
            print(f"Input file must be a JSON file: {args.input}")
    elif os.path.isdir(args.input):
        # Process all JSON files in directory
        process_folder(args.input, args.output)
    else:
        print(f"Input path does not exist: {args.input}")