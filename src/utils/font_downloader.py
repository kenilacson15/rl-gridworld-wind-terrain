"""
Download and install custom fonts for the GridWorld visualization.
This script downloads open source fonts for use in the visualization.
"""

import os
import requests
from pathlib import Path
import zipfile
import io

def download_font(url, output_dir, font_name=None):
    """Download a font file from a URL and save it to output_dir.
    
    Args:
        url: URL to download the font from
        output_dir: Directory to save the font to
        font_name: Name to save the font as (if None, extract from URL)
    
    Returns:
        Path to the downloaded font file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If font_name not provided, extract from URL
    if font_name is None:
        font_name = url.split('/')[-1]
    
    output_path = os.path.join(output_dir, font_name)
    
    # Check if font already exists
    if os.path.exists(output_path):
        print(f"Font already exists at {output_path}")
        return output_path
    
    # Download the font
    print(f"Downloading font from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    # Save the font
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    print(f"Font saved to {output_path}")
    return output_path

def download_google_font(font_name, output_dir):
    """Download a Google Font and extract the TTF/OTF files.
    
    Args:
        font_name: Name of the Google Font to download
        output_dir: Directory to save the font files to
    
    Returns:
        List of paths to the downloaded font files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Google Fonts API URL for downloading font
    url = f"https://fonts.google.com/download?family={font_name.replace(' ', '%20')}"
    
    # Download the zip file
    print(f"Downloading Google Font: {font_name}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading font: {e}")
        return []
    
    # Extract the TTF/OTF files from the zip
    font_files = []
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for filename in z.namelist():
                if filename.endswith('.ttf') or filename.endswith('.otf'):
                    # Extract the file
                    output_path = os.path.join(output_dir, os.path.basename(filename))
                    
                    # Check if file already exists
                    if not os.path.exists(output_path):
                        with open(output_path, 'wb') as f:
                            f.write(z.read(filename))
                        print(f"Extracted {filename} to {output_path}")
                    else:
                        print(f"File already exists: {output_path}")
                        
                    font_files.append(output_path)
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        return []
    
    print(f"Downloaded and extracted {len(font_files)} font files")
    return font_files

def main():
    """Download fonts for the project."""
    # Define paths relative to project root
    import os
    import sys
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fonts_dir = os.path.join(project_root, "assets", "fonts")
    
    # Create fonts directory if it doesn't exist
    os.makedirs(fonts_dir, exist_ok=True)
    
    # Download Google Fonts
    fonts_to_download = [
        "Roboto",
        "Orbitron",  # Good for sci-fi/tech UI
    ]
    
    for font in fonts_to_download:
        download_google_font(font, fonts_dir)
    
    # Alternative: Download specific OFL licensed fonts directly
    # These URLs are for demonstration - they might change or be unavailable
    font_urls = {
        "Press Start 2P": "https://fonts.google.com/download?family=Press%20Start%202P",
        "Silkscreen": "https://fonts.google.com/download?family=Silkscreen"
    }
    
    for font_name, url in font_urls.items():
        try:
            download_google_font(font_name, fonts_dir)
        except Exception as e:
            print(f"Failed to download {font_name}: {e}")
    
    print("Font download completed!")

if __name__ == "__main__":
    main()
