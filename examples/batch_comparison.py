"""
Batch comparison example for Image Similarity Toolkit.

This script demonstrates how to compare multiple images against a reference image.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_similarity import ImageSimilarity
from image_similarity.visualization import visualize_batch_results


def compare_directory(
    reference_image: str,
    comparison_dir: str,
    model_name: str = 'efficientnet'
):
    """
    Compare a reference image against all images in a directory.
    
    Args:
        reference_image: Path to the reference image
        comparison_dir: Directory containing images to compare
        model_name: Model to use for comparison
    """
    print("=" * 60)
    print("Image Similarity Toolkit - Batch Comparison Example")
    print("=" * 60)
    
    # Initialize similarity checker
    print(f"\nInitializing ImageSimilarity with {model_name}...")
    checker = ImageSimilarity(model_name=model_name)
    print(f"Using device: {checker.device}")
    
    # Get all image files from directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    comparison_images = [
        f for f in Path(comparison_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not comparison_images:
        print(f"\nNo images found in {comparison_dir}")
        return
    
    print(f"\nFound {len(comparison_images)} images to compare")
    print(f"Reference image: {reference_image}\n")
    
    # Compare each image
    results_list = []
    
    for idx, img_path in enumerate(comparison_images, 1):
        img_path_str = str(img_path)
        img_name = img_path.name
        
        print(f"[{idx}/{len(comparison_images)}] Comparing {img_name}...")
        
        try:
            results = checker.compare_images(reference_image, img_path_str)
            
            similarity = results['cosine_similarity']
            print(f"  Cosine similarity: {similarity:.4f}")
            
            results_list.append((img_name, results))
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Sort by similarity (descending)
    results_list.sort(key=lambda x: x[1]['cosine_similarity'], reverse=True)
    
    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY - Top 10 Most Similar Images")
    print("=" * 60)
    
    for idx, (name, results) in enumerate(results_list[:10], 1):
        similarity = results['cosine_similarity']
        status = checker.interpret_similarity(similarity)
        
        print(f"{idx:2d}. {name:30s} | {similarity:.4f} | {status}")
    
    # Create batch visualization
    output_path = 'data/output/batch_comparison_results.png'
    os.makedirs('data/output', exist_ok=True)
    
    print(f"\nCreating batch visualization...")
    visualize_batch_results(results_list[:10], save_path=output_path)
    print(f"Visualization saved to: {output_path}")
    
    # Save results to file
    results_file = 'data/output/batch_comparison_results.txt'
    with open(results_file, 'w') as f:
        f.write("Image Similarity Batch Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Reference Image: {reference_image}\n")
        f.write(f"Model Used: {model_name}\n")
        f.write(f"Total Images Compared: {len(results_list)}\n\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<6} {'Image Name':<30} {'Similarity':<12} {'Status'}\n")
        f.write("-" * 60 + "\n")
        
        for idx, (name, results) in enumerate(results_list, 1):
            similarity = results['cosine_similarity']
            status = checker.interpret_similarity(similarity)
            f.write(f"{idx:<6} {name:<30} {similarity:.4f}       {status}\n")
    
    print(f"Detailed results saved to: {results_file}")


def main():
    """
    Main function to run batch comparison.
    """
    # Define paths
    reference_image = 'data/input/reference.jpg'
    comparison_dir = 'data/input/compare'
    
    # Check if paths exist
    if not os.path.exists(reference_image):
        print(f"\nERROR: Reference image not found: {reference_image}")
        print("Please add a reference image to data/input/reference.jpg")
        return
    
    if not os.path.exists(comparison_dir):
        print(f"\nERROR: Comparison directory not found: {comparison_dir}")
        print("Please create data/input/compare/ and add images to compare")
        return
    
    # Run comparison
    compare_directory(
        reference_image=reference_image,
        comparison_dir=comparison_dir,
        model_name='efficientnet'
    )


if __name__ == "__main__":
    main()
