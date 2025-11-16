"""
Database indexing example for Image Similarity Toolkit.

This script demonstrates how to:
1. Index a directory of images into a database
2. Search for similar images
3. Find duplicates
"""

import sys
import os
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_similarity import ImageSimilarity, EmbeddingDatabase


def index_directory(
    image_dir: str,
    db_path: str = 'data/embeddings.db',
    model_name: str = 'efficientnet'
):
    """
    Index all images in a directory.
    
    Args:
        image_dir: Directory containing images to index
        db_path: Path to the database file
        model_name: Model to use for embeddings
    """
    print("=" * 70)
    print("Image Directory Indexing")
    print("=" * 70)
    
    # Initialize
    print(f"\nInitializing ImageSimilarity with {model_name}...")
    checker = ImageSimilarity(model_name=model_name)
    
    print(f"Opening database: {db_path}...")
    db = EmbeddingDatabase(db_path=db_path, model_name=model_name)
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = [
        f for f in Path(image_dir).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        print(f"\nNo images found in {image_dir}")
        return
    
    print(f"\nFound {len(image_paths)} images to index")
    print("-" * 70)
    
    # Index each image
    indexed = 0
    skipped = 0
    errors = 0
    
    for idx, img_path in enumerate(image_paths, 1):
        img_path_str = str(img_path)
        img_name = img_path.name
        
        print(f"[{idx}/{len(image_paths)}] Indexing: {img_name}...", end=' ')
        
        try:
            # Check if already indexed
            existing = db.get_embedding(img_path_str)
            if existing is not None:
                print("(already indexed, skipping)")
                skipped += 1
                continue
            
            # Get embedding
            embedding = checker.get_embedding(img_path_str)
            
            # Get image metadata
            try:
                img = Image.open(img_path_str)
                metadata = {
                    'file_size': img_path.stat().st_size,
                    'width': img.width,
                    'height': img.height
                }
            except:
                metadata = {}
            
            # Add to database
            db.add_image(img_path_str, embedding, metadata)
            print("✓ indexed")
            indexed += 1
            
        except Exception as e:
            print(f"✗ error: {e}")
            errors += 1
    
    print("-" * 70)
    print(f"\nIndexing complete:")
    print(f"  Indexed: {indexed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    
    # Show statistics
    stats = db.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Model: {stats['model_name']}")
    print(f"  Database: {stats['database_path']}")
    
    if stats['avg_file_size']:
        print(f"  Avg file size: {stats['avg_file_size']/1024:.1f} KB")
    if stats['avg_width']:
        print(f"  Avg dimensions: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}")
    
    db.close()
    print("\n✓ Database closed")


def search_similar_images(
    query_image: str,
    db_path: str = 'data/embeddings.db',
    model_name: str = 'efficientnet',
    top_k: int = 5,
    threshold: float = 0.5
):
    """
    Search for images similar to a query image.
    
    Args:
        query_image: Path to the query image
        db_path: Path to the database file
        model_name: Model to use
        top_k: Number of results to return
        threshold: Minimum similarity threshold
    """
    print("=" * 70)
    print("Image Similarity Search")
    print("=" * 70)
    
    # Initialize
    print(f"\nQuery image: {query_image}")
    print(f"Loading model: {model_name}...")
    checker = ImageSimilarity(model_name=model_name)
    
    print(f"Opening database: {db_path}...")
    db = EmbeddingDatabase(db_path=db_path, model_name=model_name)
    
    # Get query embedding
    print("Extracting query embedding...")
    query_embedding = checker.get_embedding(query_image)
    
    # Search
    print(f"Searching for top {top_k} similar images (threshold: {threshold:.2f})...")
    results = db.find_similar(query_embedding, top_k=top_k, threshold=threshold)
    
    if not results:
        print("\nNo similar images found!")
        db.close()
        return
    
    # Display results
    print("\n" + "=" * 70)
    print(f"Found {len(results)} similar images:")
    print("=" * 70)
    
    for idx, result in enumerate(results, 1):
        similarity = result['similarity']
        status = checker.interpret_similarity(similarity)
        
        print(f"{idx:2d}. {result['image_name']}")
        print(f"    Path: {result['image_path']}")
        print(f"    Similarity: {similarity:.4f} ({status})")
        print()
    
    db.close()


def find_duplicates_in_db(
    db_path: str = 'data/embeddings.db',
    model_name: str = 'efficientnet',
    threshold: float = 0.95
):
    """
    Find duplicate images in the database.
    
    Args:
        db_path: Path to the database file
        model_name: Model to use
        threshold: Similarity threshold for duplicates
    """
    print("=" * 70)
    print("Duplicate Detection")
    print("=" * 70)
    
    print(f"\nOpening database: {db_path}...")
    db = EmbeddingDatabase(db_path=db_path, model_name=model_name)
    
    stats = db.get_stats()
    print(f"Total images in database: {stats['total_images']}")
    print(f"Similarity threshold: {threshold:.2f}")
    
    print("\nSearching for duplicates...")
    duplicates = db.find_duplicates(
        similarity_threshold=threshold,
        save_to_table=True
    )
    
    if not duplicates:
        print("\n✓ No duplicates found!")
        db.close()
        return
    
    # Display results
    print("\n" + "=" * 70)
    print(f"Found {len(duplicates)} duplicate pairs:")
    print("=" * 70)
    
    for idx, dup in enumerate(duplicates, 1):
        print(f"\nDuplicate Pair #{idx}:")
        print(f"  Image 1: {dup['image1_name']}")
        print(f"    Path: {dup['image1_path']}")
        print(f"  Image 2: {dup['image2_name']}")
        print(f"    Path: {dup['image2_path']}")
        print(f"  Similarity: {dup['similarity']:.4f}")
    
    # Save to file
    output_file = 'data/output/duplicates_report.txt'
    os.makedirs('data/output', exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("Duplicate Images Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Database: {db_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Threshold: {threshold:.2f}\n")
        f.write(f"Total duplicates found: {len(duplicates)}\n\n")
        f.write("-" * 70 + "\n\n")
        
        for idx, dup in enumerate(duplicates, 1):
            f.write(f"Duplicate Pair #{idx}:\n")
            f.write(f"  Image 1: {dup['image1_path']}\n")
            f.write(f"  Image 2: {dup['image2_path']}\n")
            f.write(f"  Similarity: {dup['similarity']:.4f}\n\n")
    
    print(f"\n✓ Report saved to: {output_file}")
    
    db.close()


def main():
    """
    Main function with example usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Database Operations')
    parser.add_argument('operation', choices=['index', 'search', 'duplicates'],
                       help='Operation to perform')
    parser.add_argument('--dir', default='data/input',
                       help='Directory to index (for index operation)')
    parser.add_argument('--query', default='data/input/reference.jpg',
                       help='Query image (for search operation)')
    parser.add_argument('--db', default='data/embeddings.db',
                       help='Database path')
    parser.add_argument('--model', default='efficientnet',
                       choices=['resnet', 'efficientnet', 'clip'],
                       help='Model to use')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results (for search)')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Similarity threshold')
    
    args = parser.parse_args()
    
    if args.operation == 'index':
        index_directory(args.dir, args.db, args.model)
    elif args.operation == 'search':
        search_similar_images(args.query, args.db, args.model, 
                            args.top_k, args.threshold)
    elif args.operation == 'duplicates':
        find_duplicates_in_db(args.db, args.model, args.threshold)


if __name__ == "__main__":
    # Example usage without arguments
    print("Image Similarity Toolkit - Database Example\n")
    
    # Check if we have arguments
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage examples:")
        print("  python database_example.py index --dir data/input")
        print("  python database_example.py search --query image.jpg")
        print("  python database_example.py duplicates --threshold 0.95")
        print("\nRunning default demo...\n")
        
        # Demo: Index a directory if it exists
        if os.path.exists('data/input'):
            index_directory('data/input', 'data/embeddings.db', 'efficientnet')
        else:
            print("Please create 'data/input' directory and add images to try the demo.")
