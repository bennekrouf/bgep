use anyhow::Result;
use bgep::ParameterExtractor;

fn main() -> Result<()> {
    let extractor = ParameterExtractor::new()?;

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
