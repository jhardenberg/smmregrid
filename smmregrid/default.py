# default minimum area for conservative remapping
DEFAULT_AREA_MIN = 0.5 
# default threshold for skipna option: 1 is conserivative but seems to replicate CDO behavior
DEFAULT_NA_THRES = "auto"  

# Cdo generate supported methods and norms
SUPPORTED_METHODS = ["bic", "bil", "con", "con2", "dis", "laf", "nn", "ycon"]
SUPPORTED_NORM = ["fracarea", "destarea"]