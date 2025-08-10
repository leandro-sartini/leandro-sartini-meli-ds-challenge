#!/bin/bash

# Start Streamlit in the background
streamlit run src/dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true &

# Wait a moment for Streamlit to start
sleep 3

# Display user-friendly URL information
echo ""
echo "🚀 Dashboard is running!"
echo "📊 Access your dashboard at: http://localhost:8501"
echo "🔗 Or use: http://127.0.0.1:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Keep the script running
wait
