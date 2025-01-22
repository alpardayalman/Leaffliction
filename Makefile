
ZIP_URL = https://cdn.intra.42.fr/document/document/29222/leaves.zip
ZIP_FILE = leaves.zip
EXTRACT_DIR = leaves

.PHONY: all download extract clean

all: download extract

download:
	@echo "Downloading $(ZIP_FILE)..."
	@curl -O $(ZIP_URL)
	@echo "Download complete."

extract: $(ZIP_FILE)
	@echo "Extracting $(ZIP_FILE)..."
	@unzip -o $(ZIP_FILE) -d $(EXTRACT_DIR)
	@echo "Extraction complete."

clean:
	@echo "Cleaning up..."
	@rm -rf $(ZIP_FILE) $(EXTRACT_DIR)
	@echo "Clean-up complete."
