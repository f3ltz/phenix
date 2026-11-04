import matplotlib

# Render figures off-screen; the default Tk backend on Windows intermittently fails to find Tcl
matplotlib.use("Agg")
