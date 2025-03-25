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
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['module', 'width', 'l1'])
    df['module'] = df['module'].astype(str)
    widths = sorted(df['width'].unique())
    df['block'] = df['module'].str.extract(r'blocks\.(\d+)\.', expand=False)
    df['is_block'] = df['block'].notna()
    
    def get_layer_name(module):
        if 'blocks.' in module and '._orig_mod.' in module:
            return module.split('._orig_mod.')[1]
        return module
    
    df['layer_name'] = df['module'].apply(get_layer_name)
    
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
    sqrt2 = np.sqrt(2)
    
    def expected_scaling(category):
        if category == 'Input Projection':
            return sqrt2
        elif category == 'Output Projection':
            return 1/sqrt2
        else:
            return 1.0
            
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1])
    scaling_data = []
    unique_layers = df['layer_name'].unique()
    
    for layer_name in unique_layers:
        layer_data = df[df['layer_name'] == layer_name]
        
        if any(layer_data['is_block']):
            blocks = layer_data['block'].dropna().unique()
            for block in blocks:
                block_layer_data = layer_data[layer_data['block'] == block]
                
                if len(block_layer_data) < 2:
                    continue
                    
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
                        
                        scaling_data.append({
                            'block': block,
                            'layer': layer_name,
                            'category': category,
                            'width_transition': f'{prev_width}→{curr_width}',
                            'scaling_ratio': ratio,
                            'expected_ratio': expected,
                            'deviation': deviation
                        })
        else:
            if len(layer_data) < 2:
                continue
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
                    
                    scaling_data.append({
                        'block': 'N/A',
                        'layer': layer_name,
                        'category': category,
                        'width_transition': f'{prev_width}→{curr_width}',
                        'scaling_ratio': ratio,
                        'expected_ratio': expected,
                        'deviation': deviation
                    })
    
    scaling_df = pd.DataFrame(scaling_data)
    if scaling_df.empty:
        print("No valid scaling data found to analyze!")
        return None, None
    
    categories = ['Input Projection', 'Output Projection', 'Normalization', 'Embedding', 'Other']
    cat_axes = {}
    
    if 'Input Projection' in scaling_df['category'].values:
        cat_axes['Input Projection'] = fig.add_subplot(gs[0, 0])
        
    if 'Output Projection' in scaling_df['category'].values:
        cat_axes['Output Projection'] = fig.add_subplot(gs[0, 1])
    
    for cat, ax in cat_axes.items():
        if cat in scaling_df['category'].values:
            cat_scaling_df = scaling_df[scaling_df['category'] == cat]
            if not cat_scaling_df.empty and 'block' in cat_scaling_df.columns and '0' in cat_scaling_df['block'].values:
                sample_block = '0'
            elif not cat_scaling_df.empty:
                sample_block = cat_scaling_df['block'].iloc[0]
            else:
                continue
            
            cat_layers = scaling_df[
                (scaling_df['category'] == cat) & 
                (scaling_df['block'] == sample_block)
            ]['layer'].unique()
            
            if len(cat_layers) == 0:
                continue
                
            plot_data = []
            
            for layer in cat_layers:
                for width in widths:
                    layer_width_data = df[
                        (df['layer_name'] == layer) & 
                        (df['width'] == width)
                    ]
                    
                    if 'block' in scaling_df.columns and sample_block != 'N/A':
                        layer_width_data = layer_width_data[layer_width_data['block'] == sample_block]
                    
                    if len(layer_width_data) > 0:
                        plot_data.append({
                            'width': width,
                            'l1_norm': float(layer_width_data['l1'].values[0]),
                            'layer': layer
                        })
            
            if len(plot_data) > 0:
                plot_df = pd.DataFrame(plot_data)
                sns.lineplot(data=plot_df, x='width', y='l1_norm', hue='layer', marker='o', ax=ax)
                x = np.array(widths, dtype=float)
                base_val = plot_df[plot_df['width'] == widths[0]]['l1_norm'].mean()
                
                if cat == 'Input Projection':
                    y = base_val * np.sqrt(x/widths[0])
                    ax.plot(x, y, 'k--', label='Ideal (√width)')
                elif cat == 'Output Projection':
                    y = base_val * np.sqrt(widths[0]/x)
                    ax.plot(x, y, 'k--', label='Ideal (1/√width)')
                else:
                    y = np.array([base_val] * len(x))
                    ax.plot(x, y, 'k--', label='Ideal (constant)')
                
                ax.set_title(f'{cat} Scaling (Block {sample_block})')
                ax.set_xlabel('Width')
                ax.set_ylabel('L1 Norm')
                ax.set_xscale('log', base=2)
                ax.set_yscale('log', base=2)
                ax.grid(True, which="both", ls="--", alpha=0.3)
                ax.legend(title='Layer')
    
    if not scaling_df.empty:
        ax_heatmap = fig.add_subplot(gs[1, :])
        
        pivot_df = scaling_df.pivot_table(
            index=['layer', 'category', 'block'], 
            columns='width_transition', 
            values='deviation',
            aggfunc='mean'
        ).reset_index()
        
        pivot_df['sort_cat'] = pivot_df['category'].map({
            'Input Projection': 0, 
            'Output Projection': 1,
            'Normalization': 2,
            'Embedding': 3,
            'Other': 4
        })
        pivot_df = pivot_df.sort_values(['sort_cat', 'layer', 'block'])
        
        pivot_df['row_label'] = pivot_df.apply(
            lambda x: f"{x['layer']} (Block {x['block']})" if x['block'] != 'N/A' else x['layer'], 
            axis=1
        )
        width_transitions = [f'{widths[i-1]}→{widths[i]}' for i in range(1, len(widths))]
        cols_to_use = [col for col in width_transitions if col in pivot_df.columns]
        
        if cols_to_use:
            heatmap_data = pivot_df.set_index('row_label')[cols_to_use]
            vmax = min(3, np.nanmax(heatmap_data.values))
            vmin = max(-3, np.nanmin(heatmap_data.values))
            sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
                        vmin=vmin, vmax=vmax,
                        annot=True, fmt='.2f', cbar_kws={'label': 'Deviation from Ideal'},
                        ax=ax_heatmap)
            
            ax_heatmap.set_title('Deviation from Ideal muP Scaling (0 = Perfect)')
            
            category_colors = {
                'Input Projection': 'blue',
                'Output Projection': 'red',
                'Normalization': 'green',
                'Embedding': 'purple',
                'Other': 'gray'
            }
            
            for i, (_, row) in enumerate(pivot_df.iterrows()):
                ax_heatmap.add_patch(plt.Rectangle(
                    (0, i), 0.15, 1, 
                    fill=True, color=category_colors.get(row['category'], 'gray'),
                    alpha=0.3, transform=ax_heatmap.get_yaxis_transform(), clip_on=False
                ))
    
    ax_summary = fig.add_subplot(gs[2, :])

    summary_data = scaling_df.groupby(['category', 'width_transition'])['deviation'].agg(
        ['mean', lambda x: np.mean(np.abs(x)), 'count']
    ).reset_index()
    summary_data.columns = ['category', 'width_transition', 'mean_deviation', 'mean_abs_deviation', 'count']
    summary_data = summary_data[summary_data['count'] > 1]
    
    if not summary_data.empty:
        sns.barplot(x='width_transition', y='mean_abs_deviation', hue='category', data=summary_data, ax=ax_summary)
        ax_summary.set_title('Average Absolute Deviation by Category and Width Transition')
        ax_summary.set_ylabel('Mean Absolute Deviation')
        ax_summary.set_xlabel('Width Transition')
        ax_summary.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    print("\n=== muP Scaling Analysis (All Layers) ===\n")
    print("Average deviation by category:")
    category_summary = scaling_df.groupby(['category', 'width_transition'])['deviation'].agg(
        ['mean', 'std', 'min', 'max', 'count']
    )
    print(category_summary)
    print("\nWorst Scaling Deviations:")
    worst_scaling = scaling_df.sort_values(by='deviation', key=abs, ascending=False).head(10)
    print(worst_scaling[['layer', 'block', 'category', 'width_transition', 'scaling_ratio', 'expected_ratio', 'deviation']])
    if 'Output Projection' in scaling_df['category'].values:
        print("\nDetailed Analysis of Output Projections:")
        output_layers = scaling_df[scaling_df['category'] == 'Output Projection']
        for layer in output_layers['layer'].unique():
            layer_data = output_layers[output_layers['layer'] == layer]
            print(f"\n{layer}:")
            for block in layer_data['block'].unique():
                block_data = layer_data[layer_data['block'] == block]
                print(f"  Block {block}:")
                for _, row in block_data.iterrows():
                    print(f"    {row['width_transition']}: Ratio = {row['scaling_ratio']:.4f} (Expected: {row['expected_ratio']:.4f}, Deviation: {row['deviation']:.4f})")
    
    plt.tight_layout()
    plt.savefig('mup_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("\nAnalysis figure saved as 'mup_scaling_analysis.png'")
    
    return fig, scaling_df

if __name__ == "__main__":
    csv_path = "./coord_checks/mup_olmo_adamw_coord.csv"
    fig, results = analyze_mup_scaling(csv_path)
    plt.show()