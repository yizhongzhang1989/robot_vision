"""
HEIC to JPG Converter

This script converts HEIC/HEIF images to JPG format with optional resizing.

Usage:
    python tests/heic_to_jpg.py --input <input_path> --output <output_path> [--width WIDTH] [--height HEIGHT] [--quality QUALITY]

Examples:
    # Convert all HEIC images, keeping original size
    python tests/heic_to_jpg.py --input ./photos --output ./output
    
    # Resize to specific width (maintains aspect ratio)
    python tests/heic_to_jpg.py -i ./photos -o ./output --width 1920
    
    # Resize to specific dimensions
    python tests/heic_to_jpg.py -i ./photos -o ./output --width 1920 --height 1080
    
    # Convert with custom quality
    python tests/heic_to_jpg.py -i ./photos -o ./output --quality 95

Requirements:
    pip install pillow-heif pillow
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import pillow_heif  # type: ignore


def setup_heif_support():
    """Register HEIF opener with PIL"""
    pillow_heif.register_heif_opener()


def convert_heic_to_jpg(input_path, output_path, target_width=None, target_height=None, quality=90):
    """
    Convert a single HEIC image to JPG with optional resizing.
    
    Args:
        input_path (Path): Path to input HEIC file
        output_path (Path): Path to output JPG file
        target_width (int, optional): Target width in pixels
        target_height (int, optional): Target height in pixels
        quality (int): JPG quality (1-100)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open HEIC image
        img = Image.open(input_path)
        
        # Convert to RGB if necessary (HEIC might have alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if dimensions specified
        if target_width or target_height:
            original_width, original_height = img.size
            
            # Calculate new dimensions
            if target_width and target_height:
                # Both dimensions specified
                new_width = target_width
                new_height = target_height
            elif target_width:
                # Only width specified, maintain aspect ratio
                aspect_ratio = original_height / original_width
                new_width = target_width
                new_height = int(target_width * aspect_ratio)
            else:
                # Only height specified, maintain aspect ratio
                aspect_ratio = original_width / original_height
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # Resize using high-quality Lanczos resampling
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save as JPG
        img.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting {input_path.name}: {e}")
        return False


def find_heic_images(input_dir):
    """
    Find all HEIC/HEIF images in a directory.
    
    Args:
        input_dir (Path): Directory to search
        
    Returns:
        list: List of Path objects for HEIC files
    """
    heic_extensions = ['.heic', '.heif', '.HEIC', '.HEIF']
    heic_files = set()  # Use set to avoid duplicates
    
    for ext in heic_extensions:
        heic_files.update(input_dir.glob(f'*{ext}'))
    
    return sorted(list(heic_files))


def convert_directory(input_dir, output_dir, target_width=None, target_height=None, quality=90):
    """
    Convert all HEIC images in a directory to JPG.
    
    Args:
        input_dir (Path): Input directory containing HEIC files
        output_dir (Path): Output directory for JPG files
        target_width (int, optional): Target width in pixels
        target_height (int, optional): Target height in pixels
        quality (int): JPG quality (1-100)
        
    Returns:
        tuple: (successful_count, failed_count)
    """
    # Ensure input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 0, 0
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all HEIC images
    heic_files = find_heic_images(input_dir)
    
    if not heic_files:
        print(f"⚠️  No HEIC/HEIF images found in {input_dir}")
        return 0, 0
    
    print(f"📷 Found {len(heic_files)} HEIC image(s)")
    print(f"📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    if target_width or target_height:
        size_info = f"{target_width or 'auto'}x{target_height or 'auto'}"
        print(f"📐 Resize: {size_info}")
    else:
        print("📐 Resize: Keep original size")
    
    print(f"🎨 Quality: {quality}")
    print("=" * 80)
    
    # Convert each image
    successful = 0
    failed = 0
    
    for idx, heic_path in enumerate(heic_files, 1):
        # Generate output filename (replace extension with .jpg)
        output_filename = heic_path.stem + '.jpg'
        output_path = output_dir / output_filename
        
        print(f"[{idx}/{len(heic_files)}] Converting: {heic_path.name} -> {output_filename}...", end=' ')
        
        if convert_heic_to_jpg(heic_path, output_path, target_width, target_height, quality):
            print("✅")
            successful += 1
        else:
            failed += 1
    
    return successful, failed


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Convert HEIC/HEIF images to JPG format with optional resizing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/heic_to_jpg.py --input ./photos --output ./output
  python tests/heic_to_jpg.py -i ./photos -o ./output
  python tests/heic_to_jpg.py -i ./photos -o ./output --width 1920
  python tests/heic_to_jpg.py -i ./photos -o ./output --width 1920 --height 1080
  python tests/heic_to_jpg.py -i ./photos -o ./output --quality 95
        """
    )
    
    parser.add_argument(
        '--input-path', '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing HEIC images'
    )
    
    parser.add_argument(
        '--output-path', '--output', '-o',
        type=str,
        required=True,
        help='Output directory for JPG images'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Target width in pixels (maintains aspect ratio if height not specified)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Target height in pixels (maintains aspect ratio if width not specified)'
    )
    
    parser.add_argument(
        '--quality', '-q',
        type=int,
        default=90,
        choices=range(1, 101),
        metavar='[1-100]',
        help='JPG quality (1-100, default: 90)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input_path).resolve()
    output_dir = Path(args.output_path).resolve()
    
    # Setup HEIF support
    setup_heif_support()
    
    # Convert images
    print("🖼️  HEIC to JPG Converter")
    print("=" * 80)
    
    successful, failed = convert_directory(
        input_dir,
        output_dir,
        target_width=args.width,
        target_height=args.height,
        quality=args.quality
    )
    
    # Print summary
    print("=" * 80)
    print("✅ Conversion complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total: {successful + failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
