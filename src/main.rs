mod config;
mod embeddings_store;
mod matcher;
use crate::matcher::Matcher;
use anyhow::Result;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Initialize matcher
    let matcher = Matcher::new("config.yaml", "db", false).await?;

    // Example JSON
    let json = r#"{
        "request": "send a email to John",
        "recipient_email": "jd340@gmail.com",
        // "email_title": "new report",
        // "email_body": "Hi James here is the new report. best regards"
    }"#;

    let matches = matcher.match_json_holistic(json).await?;

    for endpoint_match in matches {
        println!(
            "\nEndpoint: {} (confidence: {:.4}, overall score: {:.4})",
            endpoint_match.endpoint_id,
            endpoint_match.endpoint_confidence,
            endpoint_match.overall_score
        );
        println!("Matched fields:");
        for field_match in endpoint_match.field_matches {
            println!(
                "  - {} (confidence: {:.4})",
                field_match.parameter_name, field_match.confidence
            );
        }
    }

    Ok(())
}
