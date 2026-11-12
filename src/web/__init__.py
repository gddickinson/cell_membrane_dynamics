"""Web interface for the membrane dynamics tracker (FastAPI + plain ES-module front end).

- app.py: routes (/api/info, /api/demo, /api/demo/python, POST /api/analyse) and STATIC_ENDPOINTS
- analysis.py: runs src/analysis/pipeline.py and shapes the JSON (same keys as the browser engine)
- demo_data.py: the synthetic demo stack (not real data) and run-length encoding of masks
- static/: index.html, pages and the JS ports (contour_engine.js, dynamics_engine.js, stats_engine.js, tiff_io.js)
"""
