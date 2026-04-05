#!/bin/bash
# ========================================
# ACCIDENT DETECTION SYSTEM - START SCRIPT
# ========================================

echo ""
echo "🚗 INTELLIGENT ACCIDENT DETECTION SYSTEM"
echo "=========================================="
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install flask opencv-python-headless numpy werkzeug --break-system-packages -q

echo ""
echo "🚀 Starting server..."
echo ""
echo "  ✅ Open your browser at:  http://localhost:5000"
echo "  ✅ On network:            http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "  Controls:"
echo "  - Click 'Start Webcam' for live camera"
echo "  - Upload any video file for analysis"
echo "  - Use Demo buttons to test accident detection"
echo ""
echo "  Press CTRL+C to stop"
echo ""

cd "$(dirname "$0")"
python app.py
