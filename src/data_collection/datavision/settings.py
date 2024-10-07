LOG_LEVEL: str = "INFO"

# Parser settings
USER_AGENT: str = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
# Obey robots.txt rules
ROBOTSTXT_OBEY: bool = False

ITEM_PIPELINES = {
    "data_collection.datavision.pipelines.zip_pipeline.ZipPipeline": 1,
}
