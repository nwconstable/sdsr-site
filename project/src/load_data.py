from pathlib import Path
from zipfile import ZipFile
import sys
import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GPKG_PATH = DATA_DIR / "water_nat_wetlands_inv_2009-2014.gpkg"
ZIP_PATH = DATA_DIR / "gpkg_water_nat_wetlands_inv_2009_2014.zip"
ZIP_MEMBER = GPKG_PATH.name
LAYER_NAME = "statewide_NWI"

# Had some trouble with the GeoPackage being truncated when extracted from the zip file, so this function checks for that and provides instructions for how to fix it if it is the case.
def validate_geopackage() -> None:
	if not GPKG_PATH.exists():
		raise FileNotFoundError(f"GeoPackage not found: {GPKG_PATH}")

	if not ZIP_PATH.exists():
		return

	with ZipFile(ZIP_PATH) as archive:
		expected_size = archive.getinfo(ZIP_MEMBER).file_size

	actual_size = GPKG_PATH.stat().st_size

	if actual_size != expected_size:
		raise RuntimeError(
			f"The extracted GeoPackage appears to be incomplete or corrupted. "
			f"Current size: {actual_size} bytes. Expected from zip: {expected_size} bytes. "
			f"Re-extract {ZIP_MEMBER} from {ZIP_PATH.name} with a ZIP64-capable tool such as 7-Zip."
		)

##########################################################
## Main
##########################################################

def load_gdf():
	print("Validating GeoPackage integrity...")
	validate_geopackage()
	print("Loading GeoPackage with GeoPandas...")
	return gpd.read_file(GPKG_PATH, layer=LAYER_NAME)


if __name__ == "__main__":
	gdf = load_gdf()
	print(gdf.head())
	#   attribute                 wetland_type     acres hgm_code  ... cow_class1 circ39_class    hgm_symbol                                           geometry
	# 0     PEM1A  Freshwater Emergent Wetland  5.135776   TEFLVR  ...        EM1            1  Mineral Flat  MULTIPOLYGON (((369426 5026428, 369416 5026458...
	# 1     PEM1A  Freshwater Emergent Wetland  3.194084   LOFPTH  ...        EM1            1         Lotic  MULTIPOLYGON (((369557.153 5026549.41, 369550 ...
	# 2     PEM1A  Freshwater Emergent Wetland  0.400845   LOFPTH  ...        EM1            1         Lotic  MULTIPOLYGON (((364653 5026620, 364647 5026647...
	# 3     PEM1A  Freshwater Emergent Wetland  1.641549   TEFLVR  ...        EM1            1  Mineral Flat  MULTIPOLYGON (((362966.94 5026694.963, 362967....
	# 4     PEM1C  Freshwater Emergent Wetland  1.325967   TEBAVR  ...        EM1            3    Depression  MULTIPOLYGON (((371268 5027376, 371268 5027378...

	print("Saving to CSV...")
	gdf.to_csv(DATA_DIR / "geoWetlands.csv", index=False)

	print("Done.")
	sys.exit(0)