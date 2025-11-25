import os
import sys
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import which

def check_ffmpeg():
    """Check if ffmpeg is available."""
    if which("ffmpeg") is None:
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg:")
        print("  - Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  - macOS: brew install ffmpeg")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        return False
    return True

def convert_m4a_to_mp3(input_file, output_file=None, bitrate="192k", verbose=True):
    """
    Convert a single M4A file to MP3.
    
    Args:
        input_file (str): Path to input M4A file
        output_file (str): Path to output MP3 file (optional)
        bitrate (str): Output bitrate (e.g., "128k", "192k", "320k")
        verbose (bool): Print conversion details
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        input_path = Path(input_file)
        
        # Check if input file exists
        if not input_path.exists():
            print(f"Error: Input file '{input_file}' not found")
            return False
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = input_path.with_suffix('.mp3')
        else:
            output_file = Path(output_file)
        
        if verbose:
            print(f"Converting: {input_path.name} -> {output_file.name}")
        
        # Load the M4A file
        audio = AudioSegment.from_file(str(input_path), format="m4a")
        
        # Export as MP3 with specified bitrate
        audio.export(
            str(output_file),
            format="mp3",
            bitrate=bitrate,
            parameters=["-q:a", "0"]  # Highest quality VBR
        )
        
        if verbose:
            print(f"  ✓ Successfully converted to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")
        return False

def batch_convert(input_dir, output_dir=None, bitrate="192k", recursive=False):
    """
    Convert all M4A files in a directory to MP3.
    
    Args:
        input_dir (str): Input directory containing M4A files
        output_dir (str): Output directory for MP3 files (optional)
        bitrate (str): Output bitrate
        recursive (bool): Process subdirectories recursively
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' not found")
        return
    
    # Use same directory if output not specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Find all M4A files
    if recursive:
        m4a_files = list(input_path.rglob("*.m4a")) + list(input_path.rglob("*.M4A"))
    else:
        m4a_files = list(input_path.glob("*.m4a")) + list(input_path.glob("*.M4A"))
    
    if not m4a_files:
        print(f"No M4A files found in '{input_dir}'")
        return
    
    print(f"Found {len(m4a_files)} M4A file(s) to convert")
    print(f"Output directory: {output_path}")
    print(f"Bitrate: {bitrate}")
    print("-" * 50)
    
    successful = 0
    failed = 0
    
    for m4a_file in m4a_files:
        # Preserve directory structure if recursive
        if recursive and output_dir:
            rel_path = m4a_file.relative_to(input_path)
            output_file = output_path / rel_path.with_suffix('.mp3')
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_file = output_path / m4a_file.name.replace('.m4a', '.mp3').replace('.M4A', '.mp3')
        
        if convert_m4a_to_mp3(m4a_file, output_file, bitrate, verbose=True):
            successful += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"Conversion complete: {successful} successful, {failed} failed")

def main():
    parser = argparse.ArgumentParser(
        description="Convert M4A audio files to MP3 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python m4a_to_mp3.py input.m4a
  
  # Convert with custom output name
  python m4a_to_mp3_converter.py input.m4a -o output.mp3
  
  # Convert all M4A files in a directory
  python m4a_to_mp3_converter.py -d /path/to/m4a/files
  
  # Convert with custom bitrate (320k for high quality)
  python m4a_to_mp3_converter.py input.m4a -b 320k
  
  # Recursive conversion with output directory
  python m4a_to_mp3.py -d input_dir -o output_dir -r
        """
    )
    
    parser.add_argument("input", nargs="?", help="Input M4A file")
    parser.add_argument("-o", "--output", help="Output MP3 file or directory")
    parser.add_argument("-d", "--directory", help="Convert all M4A files in directory")
    parser.add_argument("-b", "--bitrate", default="192k", 
                       choices=["128k", "192k", "256k", "320k"],
                       help="Output bitrate (default: 192k)")
    parser.add_argument("-r", "--recursive", action="store_true",
                       help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Check if ffmpeg is installed
    if not check_ffmpeg():
        sys.exit(1)
    
    # Validate arguments
    if not args.input and not args.directory:
        parser.print_help()
        sys.exit(1)
    
    if args.input and args.directory:
        print("Error: Specify either a file or directory, not both")
        sys.exit(1)
    
    try:
        # Import pydub (will fail if not installed)
        from pydub import AudioSegment
    except ImportError:
        print("Error: pydub library is not installed")
        print("Install it using: pip install pydub")
        sys.exit(1)
    
    # Process based on mode
    if args.directory:
        # Batch conversion mode
        batch_convert(args.directory, args.output, args.bitrate, args.recursive)
    else:
        # Single file conversion mode
        success = convert_m4a_to_mp3(args.input, args.output, args.bitrate)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()