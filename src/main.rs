use anyhow::{Result, Context};
use bgep::ParameterExtractor;
use std::path::Path;
use tracing::{info, error, debug};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let config_path = "config.yaml";

    info!("Attempting to load config from: {}", config_path);

    // Check if file exists
    if !Path::new(config_path).exists() {
        error!("Config file not found at path: {}", config_path);
        return Err(anyhow::anyhow!("Config file not found"));
    }
    debug!("Config file exists");

    // Read file contents
    let contents = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path))?;
    debug!("Raw config contents:\n{}", contents);

    // Try to parse as YAML
    match serde_yaml::from_str::<serde_yaml::Value>(&contents) {
        Ok(yaml) => debug!("Successfully parsed YAML:\n{:#?}", yaml),
        Err(e) => error!("Failed to parse YAML: {}", e),
    }

    // Try creating the extractor
    info!("Attempting to create ParameterExtractionSystem");
    match ParameterExtractor::new(config_path) {
        Ok(_) => info!("Successfully created ParameterExtractionSystem"),
        Err(e) => {
            error!("Failed to create ParameterExtractionSystem: {}", e);
            error!("Error details: {:#?}", e);
        }
    }

    let extractor = ParameterExtractor::new(config_path)?;

    let texts = vec![
        "send the document xxx to gg@gmail.com with title 'New document' and body 'hi James, here is the document'",
        "envoie à Jeanle the document xxx dont le title 'New document' and body 'hi James, here is the document' à l'adresse email mohamed@benek.com",
        "envoie un email à john@example.com dont le titre est Hello",
    ];

    for text in texts {
        println!("\nAnalyzing text: {}", text);
        let parameters = extractor.extract_parameters(text)?;

        println!("Extracted parameters:");
        for (param_type, value) in parameters {
            println!("  {}: {}", param_type, value);
        }
    }

    Ok(())
}
