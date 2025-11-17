"""
Very basic flask app to view all images in a directory.
"""

import argparse
from pathlib import Path
from flask import Flask, jsonify, send_file

app = Flask(__name__)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp'}
SCAN_DIR = Path.cwd()

def find_images(directory):
    """Recursively find all images in directory"""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.rglob(f'*{ext}'))
        images.extend(directory.rglob(f'*{ext.upper()}'))
    return sorted([img.relative_to(directory) for img in images])

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Image Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .controls {
            position: sticky;
            top: 0;
            background: #1a1a1a;
            padding: 20px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid #333;
            z-index: 10;
        }
        .control-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: #aaa;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #fff;
            font-size: 14px;
        }
        input[type="range"] {
            width: 100%;
            height: 6px;
            background: #444;
            border-radius: 3px;
            outline: none;
        }
        .slider-value {
            display: inline-block;
            margin-left: 10px;
            color: #fff;
            font-weight: bold;
        }
        .info {
            color: #888;
            font-size: 14px;
            margin-top: 10px;
        }
        .grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(4, 1fr);
        }
        .grid-item {
            position: relative;
            overflow: hidden;
            border-radius: 4px;
            background: #2a2a2a;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .grid-item:hover {
            transform: scale(1.05);
        }
        .grid-item img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .svg-container {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .svg-container svg {
            max-width: 100%;
            max-height: 100%;
            display: block;
        }
        .grid-item .filename {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.7);
            padding: 5px;
            font-size: 11px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>
</head>
<body>
    <div class="controls">
        <div class="control-group">
            <label>
                Grid Density: <span class="slider-value" id="density-value">4</span> columns
            </label>
            <input type="range" id="density" min="1" max="10" value="4">
        </div>
        <div class="control-group">
            <label for="filter">Regex Filter:</label>
            <input type="text" id="filter" placeholder="e.g., screenshot.*\\.png">
        </div>
        <div class="info">
            Showing <span id="count">0</span> images
        </div>
    </div>
    <div class="grid" id="grid"></div>

    <script>
        let allImages = [];

        async function loadImages() {
            const response = await fetch('/api/images');
            allImages = await response.json();
            applyFilter();
        }

        function applyFilter() {
            const filterValue = document.getElementById('filter').value;
            let filtered = allImages;

            if (filterValue) {
                try {
                    const regex = new RegExp(filterValue, 'i');
                    filtered = allImages.filter(img => regex.test(img));
                } catch (e) {
                    console.error('Invalid regex:', e);
                }
            }

            displayImages(filtered);
        }

        function displayImages(images) {
            const grid = document.getElementById('grid');
            const count = document.getElementById('count');

            grid.innerHTML = '';
            count.textContent = images.length;

            images.forEach(img => {
                const item = document.createElement('div');
                item.className = 'grid-item';

                const isSvg = img.toLowerCase().endsWith('.svg');
                const imageTag = isSvg
                    ? `<div class="svg-container" data-svg-path="${img}"></div>`
                    : `<img src="/images/${img}" alt="${img}">`;

                item.innerHTML = `
                    ${imageTag}
                    <div class="filename">${img}</div>
                `;
                item.onclick = () => window.open(`/images/${img}`, '_blank');
                grid.appendChild(item);
            });

            loadInlineSVGs();
        }

        async function loadInlineSVGs() {
            const containers = document.querySelectorAll('.svg-container[data-svg-path]');

            for (const container of containers) {
                const svgPath = container.getAttribute('data-svg-path');
                try {
                    const response = await fetch(`/images/${svgPath}`);
                    const svgText = await response.text();

                    const parser = new DOMParser();
                    const svgDoc = parser.parseFromString(svgText, 'image/svg+xml');
                    const svgElement = svgDoc.documentElement;

                    container.innerHTML = '';
                    container.appendChild(svgElement);
                } catch (error) {
                    console.error(`Failed to load SVG: ${svgPath}`, error);
                }
            }
        }

        function updateDensity() {
            const density = document.getElementById('density').value;
            document.getElementById('density-value').textContent = density;
            document.getElementById('grid').style.gridTemplateColumns =
                `repeat(${density}, 1fr)`;
        }

        document.getElementById('density').addEventListener('input', updateDensity);
        document.getElementById('filter').addEventListener('input', applyFilter);

        loadImages();
    </script>
</body>
</html>
'''

@app.route('/api/images')
def api_images():
    images = find_images(SCAN_DIR)
    return jsonify([str(img) for img in images])

@app.route('/images/<path:filepath>')
def serve_image(filepath):
    return send_file(SCAN_DIR / filepath)

def main():
    global SCAN_DIR

    parser = argparse.ArgumentParser(description='Simple local image viewer')
    parser.add_argument('directory', nargs='?', default='.',
                       help='Directory to scan for images (default: current directory)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run server on (default: 5000)')
    args = parser.parse_args()

    SCAN_DIR = Path(args.directory).resolve()

    if not SCAN_DIR.exists():
        print(f"Error: Directory '{SCAN_DIR}' does not exist")
        return

    print(f"Scanning directory: {SCAN_DIR}")
    images = find_images(SCAN_DIR)
    print(f"Found {len(images)} images")
    print(f"Starting server at http://localhost:{args.port}")

    app.run(host='127.0.0.1', port=args.port, debug=True)

if __name__ == "__main__":
    main()
