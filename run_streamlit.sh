#!/bin/bash
# Launch the PRZZ Mollifier Explorer Streamlit app

cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting PRZZ Mollifier Explorer..."
echo "Open http://localhost:8501 in your browser"
echo ""

streamlit run streamlit_app/app.py --server.port 8501
