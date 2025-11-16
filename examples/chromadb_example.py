"""
Example demonstrating ChromaDB integration with Image Similarity Toolkit.

This example shows how to use ChromaDB as an alternative backend for storing
and searching image embeddings with modern vector database capabilities.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_similarity.chromadb_backend import ChromaDBBackend


def setup_demo_images():
    """Create demo images directory with sample images."""
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)
    
    # Create a simple colored image for demo purposes
    from PIL import Image, ImageDraw
    
    # Create a simple colored square image
    for i, color in enumerate(["red", "green", "blue", "yellow"]):
        img = Image.new("RGB", (100, 100), color)
        img.save(demo_dir / f"{color}_square_{i+1}.jpg")
    
    print(f"✅ Created demo images in {demo_dir}")
    return demo_dir


def demo_basic_operations():
    """Demonstrate basic ChromaDB operations."""
    print("🚀 Basic ChromaDB Operations Demo")
    print("=" * 50)
    
    # Create demo images
    demo_dir = setup_demo_images()
    
    # Initialize ChromaDB backend
    backend = ChromaDBBackend(
        collection_name="demo_images",
        persist_directory="./demo_chroma_db"
    )
    
    # Add images from directory
    print("\n📥 Adding images to collection...")
    added_count = backend.add_images_from_directory(
        str(demo_dir),
        max_images=4,
        recursive=False
    )
    
    if added_count == 0:
        print("❌ Failed to add images")
        return
    
    # Show collection stats
    print("\n📊 Collection Statistics:")
    stats = backend.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Search by text query
    print("\n🔍 Text-based similarity search...")
    search_queries = ["red square", "colorful image", "blue shape"]
    
    for query in search_queries:
        print(f"\n--- Searching for: '{query}' ---")
        results = backend.find_similar(
            query_text=query,
            top_k=3,
            threshold=0.1
        )
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['filename']}")
                print(f"   Similarity: {result['similarity']:.3f}")
                print(f"   Path: {result['path']}")
        else:
            print("   No results found")
    
    # Search by image similarity
    print("\n🖼️  Image-based similarity search...")
    first_image = list(demo_dir.glob("*.jpg"))[0]
    print(f"Using query image: {first_image.name}")
    
    results = backend.find_similar(
        query_image_path=str(first_image),
        top_k=3
    )
    
    if results:
        print("Most similar images:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['filename']} (similarity: {result['similarity']:.3f})")
    
    # Find duplicates
    print("\n🔄 Finding duplicate images...")
    duplicates = backend.find_duplicates(similarity_threshold=0.8)
    
    if duplicates:
        for dup in duplicates:
            print(f"Duplicate pair: {dup['image1_filename']} ↔ {dup['image2_filename']}")
            print(f"   Similarity: {dup['similarity']:.3f}")
    else:
        print("No duplicates found")
    
    # Search by metadata
    print("\n🏷️  Metadata-based search...")
    jpg_results = backend.search_by_metadata(
        {"format": "JPEG"},
        limit=10
    )
    
    print(f"Found {len(jpg_results)} JPEG images")
    
    return backend


def demo_advanced_features():
    """Demonstrate advanced ChromaDB features."""
    print("\n\n🚀 Advanced ChromaDB Features Demo")
    print("=" * 50)
    
    # Initialize with custom model
    print("\n🎯 Using custom embedding model...")
    backend = ChromaDBBackend(
        collection_name="advanced_demo",
        persist_directory="./advanced_chroma_db",
        embedding_model="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create some demo images with different sizes
    from PIL import Image
    
    demo_dir = Path("advanced_demo_images")
    demo_dir.mkdir(exist_ok=True)
    
    # Create images of different sizes
    for size in [(50, 50), (200, 200), (100, 200)]:
        img = Image.new("RGB", size, "purple")
        img.save(demo_dir / f"purple_{size[0]}x{size[1]}.jpg")
    
    # Add images with custom metadata
    print("\n📝 Adding images with custom metadata...")
    for image_path in demo_dir.glob("*.jpg"):
        # Load image to get dimensions
        img = Image.open(image_path)
        
        metadata = {
            "category": "demo",
            "color": "purple",
            "aspect_ratio": img.size[0] / img.size[1],
            "demo_tag": True
        }
        
        backend.add_image(str(image_path), metadata=metadata)
    
    # Advanced similarity search with metadata filter
    print("\n🔍 Advanced search with metadata filtering...")
    results = backend.find_similar(
        query_text="purple image",
        top_k=5,
        filter_metadata={"color": "purple"}
    )
    
    if results:
        print(f"Found {len(results)} purple images:")
        for result in results:
            print(f"   {result['filename']} (similarity: {result['similarity']:.3f})")
            print(f"   Aspect ratio: {result['metadata'].get('aspect_ratio', 'N/A')}")
    
    # Export collection
    print("\n💾 Exporting collection...")
    export_file = "collection_export.json"
    if backend.export_collection(export_file):
        print(f"✅ Collection exported to {export_file}")
    
    return backend


def demo_performance_comparison():
    """Compare performance with SQLite backend."""
    print("\n\n⚡ Performance Comparison Demo")
    print("=" * 50)
    
    import time
    
    # Create larger dataset for performance testing
    from PIL import Image
    import numpy as np
    
    print("\n📊 Creating test dataset...")
    test_dir = Path("perf_test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Generate 20 test images with random colors
    np.random.seed(42)
    for i in range(20):
        # Random color image
        color = tuple(np.random.randint(0, 256, 3))
        img = Image.new("RGB", (100, 100), color)
        img.save(test_dir / f"test_image_{i:03d}.jpg")
    
    # Initialize ChromaDB backend
    print("🔧 Initializing ChromaDB backend...")
    backend = ChromaDBBackend(
        collection_name="performance_test",
        persist_directory="./perf_test_db"
    )
    
    # Test batch insertion performance
    print("📥 Testing batch insertion performance...")
    start_time = time.time()
    added_count = backend.add_images_from_directory(str(test_dir))
    insertion_time = time.time() - start_time
    
    print(f"✅ Added {added_count} images in {insertion_time:.2f} seconds")
    print(f"   Rate: {added_count/insertion_time:.1f} images/second")
    
    # Test search performance
    print("🔍 Testing search performance...")
    query_image = test_dir / "test_image_000.jpg"
    
    search_times = []
    for i in range(5):  # Run 5 searches
        start_time = time.time()
        results = backend.find_similar(
            query_image_path=str(query_image),
            top_k=10
        )
        search_time = time.time() - start_time
        search_times.append(search_time)
    
    avg_search_time = np.mean(search_times)
    print(f"✅ Average search time: {avg_search_time:.3f} seconds")
    print(f"   Found {len(results)} results in each search")
    
    # Show final statistics
    print("\n📈 Final Collection Statistics:")
    stats = backend.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")


def main():
    """Main demonstration function."""
    print("🎯 ChromaDB Integration Demo for Image Similarity Toolkit")
    print("=" * 70)
    print()
    
    print("📦 Required packages:")
    print("   pip install chromadb sentence-transformers pillow numpy")
    print()
    
    # Check if required packages are installed
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n💡 Install missing packages with:")
        print("   pip install chromadb sentence-transformers")
        return
    
    try:
        # Run basic operations demo
        backend1 = demo_basic_operations()
        
        # Run advanced features demo
        backend2 = demo_advanced_features()
        
        # Run performance comparison
        demo_performance_comparison()
        
        print("\n\n🎉 All demonstrations completed successfully!")
        print("\n💡 Next steps:")
        print("   • Try with your own image collections")
        print("   • Experiment with different embedding models")
        print("   • Explore ChromaDB persistence features")
        print("   • Integrate with your existing workflow")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 Check that all dependencies are installed correctly")


if __name__ == "__main__":
    main()