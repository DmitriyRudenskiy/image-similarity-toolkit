# Data Directory

This directory contains input and output data for image similarity comparisons.

## Structure

- `input/` - Place your input images here for comparison
- `output/` - Generated visualizations and results will be saved here

## Usage

### Input Images

Add images you want to compare to the `input/` directory:

```bash
data/input/
├── image1.jpg
├── image2.jpg
├── reference.jpg
└── compare/
    ├── compare1.jpg
    ├── compare2.jpg
    └── compare3.jpg
```

### Output Files

After running comparisons, results will be saved to `output/`:

```bash
data/output/
├── comparison_result.png          # Single comparison visualization
├── batch_comparison_results.png   # Batch comparison chart
└── batch_comparison_results.txt   # Detailed results file
```

## Notes

- Supported image formats: JPG, JPEG, PNG, BMP, WEBP
- Images are automatically converted to RGB during processing
- Large images will be resized according to model requirements
- Output files are automatically created by the toolkit
