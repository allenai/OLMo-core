import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re

def analyze_mup_scaling(csv_path):
    """
    Analyze muP scaling patterns from a CSV file containing model weights.
    Includes ALL layers found in the data.
    
    Args:
        csv_path (str): Path to the CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Drop any rows with NaN values in critical columns
    df = df.dropna(subset=['module', 'width', 'l1'])
    
    # Ensure module column is string type
    df['module'] = df['module'].astype(str)
    
    # Get unique widths
    widths = sorted(df['width'].unique())
    
    # Extract block information
    df['block'] = df['module'].str.extract(r'blocks\.(\d+)\.', expand=False)
    df['is_block'] = df['block'].notna()
    
    # Get clean layer names
    def get_layer_name(module):
        # If it's a block module, extract the deeper structure
        if 'blocks.' in module and '._orig_mod.' in module:
            return module.split('._orig_mod.')[1]
        # For embeddings and other top-level modules
        return module
    
    df['layer_name'] = df['module'].apply(get_layer_name)
    
    # Categorize layers
    def categorize_layer(layer_name):
        if any(name in layer_name for name in ['.w_q', '.w_k', '.w_v', '.w1', '.w3']):
            return 'Input Projection'
        elif any(name in layer_name for name in ['.w_out', '.w2']):
            return 'Output Projection'
        elif 'norm' in layer_name.lower():
            return 'Normalization'
        elif 'embed' in layer_name.lower():
            return 'Embedding'
        else:
            return 'Other'
            
    df['layer_category'] = df['layer_name'].apply(categorize_layer)
    
    # Get theoretical scaling factors
    sqrt2 = np.sqrt(2)
    
    # Calculate expected scaling ratio based on layer category
    def expected_scaling(category):
        if category == 'Input Projection':
            return sqrt2  # Should scale as sqrt(width)
        elif category == 'Output Projection':
            return 1/sqrt2  # Should scale as 1/sqrt(width)
        else:
            return 1.0  # Should remain roughly constant
    
    # Define correct muP scaling standards from the paper
    def mup_correct_scaling(category, layer_name):
        """Return the correct scaling for muP based on layer type"""
        # Attention scaling
        if 'attn' in layer_name and any(x in layer_name for x in ['.w_q', '.w_k', '.w_v']):
            return 'Scale by 1/d_head instead of 1/√d_head for attention, scale weights by 1/√m_d'
        # Input projections
        elif category == 'Input Projection':
            return 'Scale initialization variance by 1/m_d'
        # Output projections
        elif category == 'Output Projection':
            return 'Scale initialization variance by 1/m_d, scale output by 1/m_d'
        # Embeddings
        elif category == 'Embedding':
            return 'Keep same initialization as standard parameterization'
        # Normalization
        elif category == 'Normalization':
            return 'No additional corrections needed'
        else:
            return 'Specific scaling depends on component'
            
    # Create figure
    fig = plt.figure(figsize=(16, 24))  # Made taller to accommodate new section
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1.5, 1, 1.5])  # Added an extra row
    
    # 1. Calculate scaling ratios for all layers
    scaling_data = []
    
    # Get all unique layer names (filtering out duplicates like module vs module.dropout)
    unique_layers = df['layer_name'].unique()
    
    # Calculate block-wise scaling for block-related modules
    for layer_name in unique_layers:
        layer_data = df[df['layer_name'] == layer_name]
        
        # For block items, separate by block
        if any(layer_data['is_block']):
            blocks = layer_data['block'].dropna().unique()
            for block in blocks:
                block_layer_data = layer_data[layer_data['block'] == block]
                
                # Skip if insufficient data points
                if len(block_layer_data) < 2:
                    continue
                    
                # Calculate scaling across width transitions
                for i in range(1, len(widths)):
                    prev_width = widths[i-1]
                    curr_width = widths[i]
                    prev_l1 = block_layer_data[block_layer_data['width'] == prev_width]['l1'].values
                    curr_l1 = block_layer_data[block_layer_data['width'] == curr_width]['l1'].values
                    
                    if len(prev_l1) > 0 and len(curr_l1) > 0:
                        ratio = float(curr_l1[0]) / float(prev_l1[0])
                        category = block_layer_data['layer_category'].iloc[0]
                        expected = expected_scaling(category)
                        deviation = ratio / expected - 1
                        
                        # Calculate width multiplier
                        width_multiplier = float(curr_width) / float(prev_width)
                        
                        # Determine if scaling is correct according to muP principles
                        is_correct = False
                        tolerance = 0.1  # 10% tolerance
                        
                        # For Input Projections, should scale with 1/√m_d
                        if category == 'Input Projection':
                            expected_mup = 1.0 / np.sqrt(width_multiplier)
                            is_correct = abs(ratio / expected_mup - 1) < tolerance
                        # For Output Projections, should scale with 1/√m_d and also need 1/m_d output scaling
                        elif category == 'Output Projection':
                            expected_mup = 1.0 / np.sqrt(width_multiplier)
                            is_correct = abs(ratio / expected_mup - 1) < tolerance
                        # For Normalization, no scaling needed
                        elif category == 'Normalization':
                            is_correct = abs(ratio - 1.0) < tolerance
                        # For Embeddings, standard initialization
                        elif category == 'Embedding':
                            is_correct = abs(ratio - 1.0) < tolerance
                        
                        scaling_data.append({
                            'block': block,
                            'layer': layer_name,
                            'category': category,
                            'width_transition': f'{prev_width}→{curr_width}',
                            'scaling_ratio': ratio,
                            'expected_ratio': expected,
                            'deviation': deviation,
                            'is_correct_mup': is_correct,
                            'mup_guidance': mup_correct_scaling(category, layer_name),
                            'width_multiplier': width_multiplier
                        })
        else:
            # For non-block items (like embeddings)
            if len(layer_data) < 2:
                continue
                
            # Calculate scaling across width transitions
            for i in range(1, len(widths)):
                prev_width = widths[i-1]
                curr_width = widths[i]
                prev_l1 = layer_data[layer_data['width'] == prev_width]['l1'].values
                curr_l1 = layer_data[layer_data['width'] == curr_width]['l1'].values
                
                if len(prev_l1) > 0 and len(curr_l1) > 0:
                    ratio = float(curr_l1[0]) / float(prev_l1[0])
                    category = layer_data['layer_category'].iloc[0]
                    expected = expected_scaling(category)
                    deviation = ratio / expected - 1
                    
                    # Calculate width multiplier
                    width_multiplier = float(curr_width) / float(prev_width)
                    
                    # Determine if scaling is correct according to muP principles
                    is_correct = False
                    tolerance = 0.1  # 10% tolerance
                    
                    if category == 'Embedding':
                        is_correct = abs(ratio - 1.0) < tolerance
                    
                    scaling_data.append({
                        'block': 'N/A',
                        'layer': layer_name,
                        'category': category,
                        'width_transition': f'{prev_width}→{curr_width}',
                        'scaling_ratio': ratio,
                        'expected_ratio': expected,
                        'deviation': deviation,
                        'is_correct_mup': is_correct,
                        'mup_guidance': mup_correct_scaling(category, layer_name),
                        'width_multiplier': width_multiplier
                    })
    
    scaling_df = pd.DataFrame(scaling_data)
    
    if scaling_df.empty:
        print("No valid scaling data found to analyze!")
        return None, None
    
    # Rest of the original code (plots, etc.)...
    # [Original visualization code continues here]
    
    # Add new section to highlight correct/incorrect muP scaling
    ax_correctness = fig.add_subplot(gs[3, :])
    
    # Calculate correctness by category
    correctness_data = scaling_df.groupby(['category'])['is_correct_mup'].agg(
        ['mean', 'count']
    ).reset_index()
    correctness_data.columns = ['Category', 'Correctness Rate', 'Count']
    
    # Sort by correctness rate
    correctness_data = correctness_data.sort_values(by='Correctness Rate', ascending=False)
    
    # Plot correctness rates
    bars = sns.barplot(x='Category', y='Correctness Rate', data=correctness_data, ax=ax_correctness)
    ax_correctness.set_title('muP Scaling Correctness by Layer Category')
    ax_correctness.set_ylabel('Proportion of Correct Scaling (within 10% tolerance)')
    ax_correctness.set_ylim(0, 1)
    
    # Add count labels on the bars
    for i, p in enumerate(bars.patches):
        bars.annotate(f"n={correctness_data.iloc[i]['Count']}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'bottom', xytext = (0, 5),
                     textcoords = 'offset points')
    
    # Add reference to correct scaling table
    mup_table = """
    Layer Type             | Correct muP Scaling
    -----------------------|---------------------------
    Embedding Init Var     | σ²_base (same as standard)
    Embedding LR           | η_base (same as standard)
    Embedding Forward      | α_input · xW_emb
    Hidden Init Var        | σ²_base/m_d
    Hidden LR (Adam)       | η_base/m_d
    Output Logit Forward   | α_output · xW⊤_emb/m_d
    Attention Logits       | Q⊤K/d_head (not 1/√d_head)
    """
    
    # Add table as text annotation
    fig.text(0.1, 0.02, mup_table, fontsize=10, family='monospace')
    
    # Print summary statistics with correctness
    print("\n=== muP Scaling Analysis (All Layers) ===\n")
    
    # Add muP correctness summary
    print("\nmuP Scaling Correctness by Category:")
    print(correctness_data)
    
    # Show layers with incorrect scaling
    incorrect_layers = scaling_df[~scaling_df['is_correct_mup']]
    if not incorrect_layers.empty:
        print("\nLayers with Incorrect muP Scaling:")
        for cat in incorrect_layers['category'].unique():
            cat_incorrect = incorrect_layers[incorrect_layers['category'] == cat]
            print(f"\n{cat} Layers:")
            for layer in cat_incorrect['layer'].unique():
                layer_incorrect = cat_incorrect[cat_incorrect['layer'] == layer]
                print(f"  {layer}:")
                for _, row in layer_incorrect.iterrows():
                    print(f"    Block {row['block']}, {row['width_transition']}: Ratio = {row['scaling_ratio']:.4f}, muP Guidance: {row['mup_guidance']}")
    
    # [Rest of original code continues...]
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('mup_scaling_analysis2.png', dpi=300, bbox_inches='tight')
    print("\nAnalysis figure saved as 'mup_scaling_analysis2.png'")
    
    return fig, scaling_df

if __name__ == "__main__":
    # Replace with your actual CSV file path
    csv_path = "./coord_checks/mup_olmo_adamw_coord.csv"
    fig, results = analyze_mup_scaling(csv_path)
    plt.show()