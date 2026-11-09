# default na_thres: 'auto' is the smmregrid default of each method, 'cdo' replicates CDO (see resolve_na_thres)
DEFAULT_NA_THRES = "auto"
# methods for which the covered area of a target cell is a fraction between 0 and 1
CONSERVATIVE_METHODS = ["con", "con2", "ycon"]

# Cdo generate supported methods and norms
SUPPORTED_METHODS = ["bic", "bil", "con", "con2", "dis", "laf", "nn", "ycon"]
SUPPORTED_NORM = ["fracarea", "destarea"]
# CDO method from the map_method attribute of the weights
CDO_MAP_METHODS = {
    "Bilinear remapping": "bil", "Bicubic remapping": "bic", "Conservative remapping": "con",
    "Nearest neighbor": "nn", "Distance weighted avg of nearest neighbors": "dis", "Largest area fraction": "laf",
}

# maximum time (s) for a CDO weights generation, to avoid hanging forever
CDO_TIMEOUT = 600

# decimals of lat/lon coordinates, shared by regridded data and target areas to keep them aligned
COORD_DECIMALS = 10
