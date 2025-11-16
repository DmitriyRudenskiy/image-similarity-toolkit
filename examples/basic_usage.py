"""
Basic usage example for Image Similarity Toolkit.

This script demonstrates how to compare two images using different models.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_similarity import ImageSimilarity


def main():
    """
    Basic example of comparing two images.
    """
    print("=" * 60)
    print("Image Similarity Toolkit - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the similarity checker with EfficientNet
    print("\nInitializing ImageSimilarity with EfficientNet-B0...")
    similarity_checker = ImageSimilarity(model_name='efficientnet')
    print(f"Using device: {similarity_checker.device}")
    
    # Define paths to images (you need to provide your own images)
    image1 = 'data/input/image1.jpg'
    image2 = 'data/input/image2.jpg'
    
    # Check if images exist
    if not os.path.exists(image1) or not os.path.exists(image2):
        print("\n" + "!" * 60)
        print("ERROR: Sample images not found!")
        print("Please add images to data/input/ directory:")
        print(f"  - {image1}")
        print(f"  - {image2}")
        print("!" * 60)
        return
    
    try:
        # Compare images
        print(f"\nComparing images:")
        print(f"  Image 1: {image1}")
        print(f"  Image 2: {image2}")
        print("\nProcessing...")
        
        results = similarity_checker.compare_images(image1, image2)
        
        # Display results
        print("\n" + "-" * 60)
        print("RESULTS:")
        print("-" * 60)
        print(f"Cosine Similarity:      {results['cosine_similarity']:.4f}")
        print(f"Euclidean Distance:     {results['euclidean_distance']:.4f}")
        print(f"Normalized Similarity:  {results['normalized_similarity']:.4f}")
        print("-" * 60)
        
        # Interpretation
        interpretation = similarity_checker.interpret_similarity(
            results['cosine_similarity']
        )
        print(f"\nInterpretation: {interpretation}")
        
        # Detailed interpretation
        print("\nDetailed Analysis:")
        if results['cosine_similarity'] > 0.85:
            print("  Status: Very similar images")
            print("  These images are nearly identical or show the same object")
        elif results['cosine_similarity'] > 0.7:
            print("  Status: Moderately similar images")
            print("  These images share common features or themes")
        elif results['cosine_similarity'] > 0.5:
            print("  Status: Weakly similar images")
            print("  These images have some visual similarities")
        else:
            print("  Status: Different images")
            print("  These images are quite different from each other")
        
        # Visualize results
        output_path = 'data/output/comparison_result.png'
        os.makedirs('data/output', exist_ok=True)
        
        print(f"\nCreating visualization...")
        similarity_checker.visualize_comparison(
            image1,
            image2,
            results,
            save_path=output_path
        )
        print(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
