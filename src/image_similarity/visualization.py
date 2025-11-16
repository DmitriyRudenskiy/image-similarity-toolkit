"""
Visualization module for image similarity comparison results.
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from typing import Dict, Optional
import warnings


def setup_matplotlib_for_plotting():
    """
    Setup matplotlib for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    warnings.filterwarnings('default')
    
    # Configure matplotlib for non-interactive mode
    matplotlib.use('Agg')
    plt.switch_backend("Agg")
    
    # Set chart style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    
    # Configure platform-appropriate fonts for cross-platform compatibility
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", 
        "WenQuanYi Zen Hei", 
        "PingFang SC", 
        "Arial Unicode MS", 
        "Hiragino Sans GB"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def visualize_comparison(
    image_path1: str,
    image_path2: str,
    results: Dict[str, float],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the comparison results between two images.
    
    Args:
        image_path1: Path to the first image
        image_path2: Path to the second image
        results: Dictionary containing comparison results with keys:
            - cosine_similarity
            - euclidean_distance
            - normalized_similarity
        save_path: Optional path to save the visualization
        
    Example:
        >>> results = {'cosine_similarity': 0.92, 'euclidean_distance': 12.5, 
        ...            'normalized_similarity': 0.88}
        >>> visualize_comparison('img1.jpg', 'img2.jpg', results, 'output.png')
    """
    # Setup matplotlib
    setup_matplotlib_for_plotting()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Load and display images
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')
    
    axes[0].imshow(img1)
    axes[0].set_title(f'Image 1\n{os.path.basename(image_path1)}', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title(f'Image 2\n{os.path.basename(image_path2)}', fontsize=12)
    axes[1].axis('off')
    
    # Add metrics text
    metrics_text = (
        f"Cosine Similarity: {results['cosine_similarity']:.4f}\n"
        f"Euclidean Distance: {results['euclidean_distance']:.4f}\n"
        f"Normalized Similarity: {results['normalized_similarity']:.4f}"
    )
    
    # Color code based on similarity
    if results['cosine_similarity'] > 0.85:
        box_color = 'lightgreen'
    elif results['cosine_similarity'] > 0.7:
        box_color = 'lightyellow'
    elif results['cosine_similarity'] > 0.5:
        box_color = 'orange'
    else:
        box_color = 'lightcoral'
    
    plt.figtext(
        0.5, 0.02,
        metrics_text,
        ha="center",
        fontsize=11,
        bbox={"facecolor": box_color, "alpha": 0.7, "pad": 8}
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {save_path}")
    
    plt.close()


def visualize_batch_results(
    results_list: list,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize multiple comparison results as a bar chart.
    
    Args:
        results_list: List of tuples (image_pair_name, results_dict)
        save_path: Optional path to save the visualization
        
    Example:
        >>> results_list = [
        ...     ('cat1_vs_cat2', {'cosine_similarity': 0.92}),
        ...     ('cat1_vs_dog1', {'cosine_similarity': 0.45})
        ... ]
        >>> visualize_batch_results(results_list, 'batch_results.png')
    """
    setup_matplotlib_for_plotting()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [item[0] for item in results_list]
    similarities = [item[1]['cosine_similarity'] for item in results_list]
    
    bars = ax.barh(names, similarities)
    
    # Color code bars
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        if sim > 0.85:
            bar.set_color('green')
        elif sim > 0.7:
            bar.set_color('yellow')
        elif sim > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_title('Image Similarity Comparison Results', fontsize=14)
    ax.set_xlim(0, 1)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Batch visualization saved to: {save_path}")
    
    plt.close()
