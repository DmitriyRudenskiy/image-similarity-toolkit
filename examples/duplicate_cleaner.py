"""
Duplicate cleaner example for Image Similarity Toolkit.

This script provides tools for managing duplicate images:
1. Detect duplicates
2. Review duplicates
3. Automatically remove duplicates
"""

import sys
import os
from pathlib import Path
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_similarity import ImageSimilarity, EmbeddingDatabase


class DuplicateCleaner:
    """
    Utility class for finding and managing duplicate images.
    """
    
    def __init__(
        self,
        db_path: str = 'data/embeddings.db',
        model_name: str = 'efficientnet',
        threshold: float = 0.95
    ):
        """
        Initialize the duplicate cleaner.
        
        Args:
            db_path: Path to the database
            model_name: Model to use for embeddings
            threshold: Similarity threshold for duplicates
        """
        self.db_path = db_path
        self.model_name = model_name
        self.threshold = threshold
        self.checker = ImageSimilarity(model_name=model_name)
        self.db = EmbeddingDatabase(db_path=db_path, model_name=model_name)
    
    def scan_and_index(self, directory: str):
        """
        Scan a directory and index all images.
        
        Args:
            directory: Directory to scan
        """
        print(f"Scanning directory: {directory}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = [
            f for f in Path(directory).rglob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_paths)} images")
        
        for idx, img_path in enumerate(image_paths, 1):
            print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}...", end=' ')
            
            try:
                # Check if already indexed
                if self.db.get_embedding(str(img_path)):
                    print("(cached)")
                    continue
                
                # Get embedding and index
                embedding = self.checker.get_embedding(str(img_path))
                
                from PIL import Image
                img = Image.open(img_path)
                metadata = {
                    'file_size': img_path.stat().st_size,
                    'width': img.width,
                    'height': img.height
                }
                
                self.db.add_image(str(img_path), embedding, metadata)
                print("✓")
                
            except Exception as e:
                print(f"✗ {e}")
    
    def find_duplicates(self):
        """
        Find all duplicate images.
        
        Returns:
            List of duplicate pairs
        """
        print(f"Finding duplicates (threshold: {self.threshold})...")
        duplicates = self.db.find_duplicates(
            similarity_threshold=self.threshold,
            save_to_table=True
        )
        
        print(f"Found {len(duplicates)} duplicate pairs")
        return duplicates
    
    def review_duplicates(self, duplicates):
        """
        Interactive review of duplicates.
        
        Args:
            duplicates: List of duplicate pairs
        """
        print("\n" + "=" * 70)
        print("Duplicate Review")
        print("=" * 70)
        
        kept = []
        removed = []
        
        for idx, dup in enumerate(duplicates, 1):
            print(f"\nDuplicate Pair #{idx}/{len(duplicates)}")
            print("-" * 70)
            
            img1_path = dup['image1_path']
            img2_path = dup['image2_path']
            
            print(f"Image 1: {img1_path}")
            print(f"Image 2: {img2_path}")
            print(f"Similarity: {dup['similarity']:.4f}")
            
            # Get file sizes
            try:
                size1 = Path(img1_path).stat().st_size
                size2 = Path(img2_path).stat().st_size
                print(f"\nFile sizes:")
                print(f"  Image 1: {size1/1024:.1f} KB")
                print(f"  Image 2: {size2/1024:.1f} KB")
            except:
                pass
            
            print("\nOptions:")
            print("  1 - Keep Image 1, remove Image 2")
            print("  2 - Keep Image 2, remove Image 1")
            print("  s - Skip (keep both)")
            print("  q - Quit review")
            
            choice = input("\nYour choice: ").lower().strip()
            
            if choice == '1':
                removed.append(img2_path)
                kept.append(img1_path)
                print(f"✓ Will remove: {img2_path}")
            elif choice == '2':
                removed.append(img1_path)
                kept.append(img2_path)
                print(f"✓ Will remove: {img1_path}")
            elif choice == 's':
                print("Skipped")
            elif choice == 'q':
                print("Quitting review...")
                break
            else:
                print("Invalid choice, skipping...")
        
        return kept, removed
    
    def auto_remove_duplicates(
        self,
        duplicates,
        strategy: str = 'smaller',
        backup_dir: str = 'data/duplicates_backup'
    ):
        """
        Automatically remove duplicates using a strategy.
        
        Args:
            duplicates: List of duplicate pairs
            strategy: 'smaller' (keep larger), 'larger' (keep smaller), 
                     'first' (keep first path)
            backup_dir: Directory to backup removed files
        """
        print(f"\nAuto-removing duplicates (strategy: {strategy})...")
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        removed = []
        
        for dup in duplicates:
            img1_path = dup['image1_path']
            img2_path = dup['image2_path']
            
            # Determine which to remove
            to_remove = None
            
            if strategy == 'smaller':
                try:
                    size1 = Path(img1_path).stat().st_size
                    size2 = Path(img2_path).stat().st_size
                    to_remove = img1_path if size1 < size2 else img2_path
                except:
                    to_remove = img2_path  # Default
                    
            elif strategy == 'larger':
                try:
                    size1 = Path(img1_path).stat().st_size
                    size2 = Path(img2_path).stat().st_size
                    to_remove = img1_path if size1 > size2 else img2_path
                except:
                    to_remove = img2_path  # Default
                    
            elif strategy == 'first':
                to_remove = img2_path
            
            if to_remove and os.path.exists(to_remove):
                # Backup
                backup_path = Path(backup_dir) / Path(to_remove).name
                try:
                    shutil.copy2(to_remove, backup_path)
                    os.remove(to_remove)
                    removed.append(to_remove)
                    print(f"✓ Removed: {to_remove}")
                    print(f"  Backup: {backup_path}")
                    
                    # Remove from database
                    self.db.remove_image(to_remove)
                    
                except Exception as e:
                    print(f"✗ Error removing {to_remove}: {e}")
        
        print(f"\n✓ Removed {len(removed)} duplicate images")
        print(f"  Backups saved to: {backup_dir}")
        
        return removed
    
    def generate_report(self, duplicates, output_file: str = 'data/output/duplicates_report.html'):
        """
        Generate an HTML report of duplicates.
        
        Args:
            duplicates: List of duplicate pairs
            output_file: Output HTML file path
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Duplicate Images Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .duplicate-pair { 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .images { display: flex; gap: 20px; margin-top: 10px; }
        .image-container { flex: 1; }
        .image-container img { max-width: 100%; border-radius: 4px; }
        .similarity { 
            font-size: 18px; 
            font-weight: bold; 
            color: #e74c3c;
            margin: 10px 0;
        }
        .info { color: #666; font-size: 14px; }
        .stats { 
            background: white; 
            padding: 20px; 
            margin-bottom: 20px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <h1>Duplicate Images Report</h1>
    <div class="stats">
        <h2>Statistics</h2>
        <p>Total duplicate pairs found: {total_duplicates}</p>
        <p>Model used: {model}</p>
        <p>Similarity threshold: {threshold:.2f}</p>
    </div>
""".format(
            total_duplicates=len(duplicates),
            model=self.model_name,
            threshold=self.threshold
        )
        
        for idx, dup in enumerate(duplicates, 1):
            html += f"""
    <div class="duplicate-pair">
        <h3>Duplicate Pair #{idx}</h3>
        <div class="similarity">Similarity: {dup['similarity']:.4f}</div>
        <div class="images">
            <div class="image-container">
                <h4>Image 1</h4>
                <p class="info">{dup['image1_path']}</p>
            </div>
            <div class="image-container">
                <h4>Image 2</h4>
                <p class="info">{dup['image2_path']}</p>
            </div>
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✓ HTML report generated: {output_file}")
    
    def close(self):
        """Close the database connection."""
        self.db.close()


def main():
    """Main function."""
    print("=" * 70)
    print("Duplicate Image Cleaner")
    print("=" * 70)
    
    # Configuration
    image_dir = 'data/input'
    db_path = 'data/embeddings.db'
    model_name = 'efficientnet'
    threshold = 0.95
    
    if not os.path.exists(image_dir):
        print(f"\nError: Directory '{image_dir}' not found!")
        print("Please create the directory and add images to scan.")
        return
    
    # Initialize cleaner
    cleaner = DuplicateCleaner(
        db_path=db_path,
        model_name=model_name,
        threshold=threshold
    )
    
    # Scan and index
    print("\nStep 1: Scanning and indexing images...")
    cleaner.scan_and_index(image_dir)
    
    # Find duplicates
    print("\nStep 2: Finding duplicates...")
    duplicates = cleaner.find_duplicates()
    
    if not duplicates:
        print("\n✓ No duplicates found!")
        cleaner.close()
        return
    
    # Generate report
    print("\nStep 3: Generating report...")
    cleaner.generate_report(duplicates)
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("  1 - Review duplicates interactively")
    print("  2 - Auto-remove duplicates (keep larger files)")
    print("  3 - Auto-remove duplicates (keep smaller files)")
    print("  4 - Exit without removing")
    
    choice = input("\nYour choice: ").strip()
    
    if choice == '1':
        kept, removed = cleaner.review_duplicates(duplicates)
        if removed:
            print(f"\nRemoving {len(removed)} images...")
            # Implement removal here if needed
            
    elif choice == '2':
        cleaner.auto_remove_duplicates(duplicates, strategy='smaller')
        
    elif choice == '3':
        cleaner.auto_remove_duplicates(duplicates, strategy='larger')
    
    else:
        print("\nExiting without removing duplicates.")
    
    cleaner.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
