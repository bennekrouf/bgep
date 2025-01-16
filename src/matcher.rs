use crate::config::Config;
use crate::embeddings_store::{EmbeddingsStore, SearchResult};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::Api;
use serde_json::Value;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info};

#[derive(Debug)]
pub struct EndpointMatches {
    pub endpoint_id: String,
    pub endpoint_confidence: f32,
    pub field_matches: Vec<FieldMatch>,
    pub overall_score: f32,
}

pub struct Matcher {
    config: Config,
    embeddings_store: EmbeddingsStore,
    model: Arc<BertModel>,
    tokenizer: Tokenizer,
    device: Device,
}

#[derive(Debug)]
pub struct FieldMatch {
    pub parameter_name: String,
    #[warn(dead_code)]
    endpoint_id: String,
    pub confidence: f32,
}

impl Matcher {
    pub async fn new(config_path: &str, db_path: &str, force_init: bool) -> Result<Self> {
        let config = Config::load_from_yaml(config_path)?;
        let embeddings_store = EmbeddingsStore::new(db_path).await?;

        // Initialize BERT model
        let device = Device::cuda_if_available(0)?;
        info!("Using device: {:?}", device);

        let api = Api::new()?;
        let model_id = "BAAI/bge-large-en";
        let repo = api.model(model_id.to_string());

        let model_path = repo.get("model.safetensors")?;
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        info!("Loading BGE-Large-EN model...");

        let bert_config: BertConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        let vb = VarBuilder::from_tensors(
            candle_core::safetensors::load(model_path, &device)?,
            DType::F32,
            &device,
        );

        let model = Arc::new(BertModel::load(vb, &bert_config)?);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let matcher = Self {
            config,
            embeddings_store,
            model,
            tokenizer,
            device,
        };

        // Only initialize embeddings if forced or if database is empty
        if force_init || matcher.embeddings_store.is_empty().await? {
            info!("Initializing embeddings database...");
            matcher.initialize_embeddings().await?;
        } else {
            info!("Using existing embeddings database");
        }

        Ok(matcher)
    }

    async fn initialize_embeddings(&self) -> Result<()> {
        for endpoint in &self.config.endpoints {
            debug!("Storing embeddings for endpoint: {}", endpoint.id);
            debug!("  Text: '{}'", endpoint.text);
            debug!("  Description: '{}'", endpoint.description);

            // Add endpoint text and description embeddings
            let text_embedding = self.compute_embedding(&endpoint.text)?;
            self.embeddings_store
                .store_embedding(
                    &format!("endpoint:{}", endpoint.id),
                    &endpoint.text,  // This is what will be compared
                    &endpoint.id,
                    text_embedding,
                )
                .await?;

            let desc_embedding = self.compute_embedding(&endpoint.description)?;
            self.embeddings_store
                .store_embedding(
                    &format!("endpoint:{}_desc", endpoint.id),
                    &endpoint.description,
                    &endpoint.id,
                    desc_embedding,
                )
                .await?;

                for param in &endpoint.parameters {
                    // Store just the parameter name
                    let param_embedding = self.compute_embedding(&param.name)?;
                    self.embeddings_store
                        .store_embedding(
                            &format!("param:{}:{}", endpoint.id, param.name),
                            &param.name,
                            &endpoint.id,
                            param_embedding,
                        )
                        .await?;

                    // Store alternatives as simple field names
                    if let Some(alternatives) = &param.alternatives {
                        for alt in alternatives {
                            let alt_embedding = self.compute_embedding(alt)?;
                            self.embeddings_store
                                .store_embedding(
                                    &format!("param:{}:{}_alt:{}", endpoint.id, param.name, alt),
                                    alt,
                                    &endpoint.id,
                                    alt_embedding,
                                )
                                .await?;
                        }
                    }
            }
        }
        Ok(())
    }

    fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let processed_text = format!("Represent this text for retrieval: {}", text);

        let tokens = self
            .tokenizer
            .encode(processed_text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?;
        let attention_mask = Tensor::new(tokens.get_attention_mask(), &self.device)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?;

        let embeddings = self.model.forward(
            &input_ids.unsqueeze(0)?,
            &attention_mask.unsqueeze(0)?,
            Some(&token_type_ids.unsqueeze(0)?),
        )?;

        // Convert attention mask to f32 and reshape
        let attention_mask = attention_mask
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .broadcast_as(embeddings.shape())?;

        let mean = embeddings.mul(&attention_mask)?;
        let sum = mean.sum(1)?;
        let counts = attention_mask.sum(1)?;
        let embedding = sum.broadcast_div(&counts)?;

        // Normalize the embedding
        let norm = embedding.sqr()?.sum_all()?.sqrt()?;
        let normalized = embedding.broadcast_div(&norm)?;

        // Remove batch dimension and convert to Vec<f32>
        let final_embedding = normalized.squeeze(0)?;
        let embedding_vec: Vec<f32> = final_embedding.to_vec1()?;

        Ok(embedding_vec)
    }

    pub async fn match_json_holistic(&self, json_str: &str) -> Result<Vec<EndpointMatches>> {

        let json: Value = serde_json::from_str(json_str)?;
        let mut endpoint_matches = Vec::new();

        // Phase 1: Match request with endpoint text/description
        let request = json["request"].as_str()
            .ok_or_else(|| anyhow::anyhow!("No request field found"))?;
        debug!("\nAnalyzing request: '{}'", request);
        let request_embedding = self.compute_embedding(request)?;

        // Get endpoint matches and deduplicate
        let mut seen_keys = std::collections::HashSet::new();
        let potential_endpoints: Vec<_> = self.embeddings_store
            .search_similar(request_embedding, 10)
            .await?
            .into_iter()
            .filter(|m| {
                let is_endpoint = m.key.starts_with("endpoint:");
                let is_pure_endpoint = !m.key.ends_with("_desc");
                debug!("Endpoint matching:");
                debug!("  Request: '{}'", request);
                debug!("  Endpoint key: '{}'", m.key);
                debug!("  Endpoint text: '{}'", m.value);  // This should be the endpoint text or description
                debug!("  Score: {:.4}", m.score);
                is_endpoint && is_pure_endpoint && seen_keys.insert(m.key.clone())
            })
            .collect();

        // Phase 2: For each potential endpoint, evaluate field matches
        for endpoint_match in potential_endpoints {
        let endpoint_id = &endpoint_match.endpoint_id;

        // Get the endpoint configuration
        let endpoint_config = self.config.endpoints.iter()
            .find(|e| &e.id == endpoint_id)
            .ok_or_else(|| anyhow::anyhow!("Endpoint not found"))?;

        // let mut endpoint_specific_matches = Vec::new();
        let endpoint_specific_matches: Vec<(&String, Vec<SearchResult>)> = Vec::new();

        // Get all field matches for this endpoint first
        if let Value::Object(map) = &json {
            for field_name in map.keys() {
                if field_name != "request" {
                    debug!("\nParameter matching:");
                    debug!("  Field name: '{}'", field_name);
                    let field_embedding = self.compute_embedding(field_name)?;
                    let matches = self.embeddings_store
                        .search_similar(field_embedding, 3)
                        .await?;

                    for m in &matches {
                        debug!("  Against parameter: '{}'", extract_parameter_name(&m.key));
                        if m.key.contains("_alt:") {
                            debug!("    (alternative name for '{}')", 
                                extract_base_param_name(&m.key));
                        }
                        debug!("    Score: {:.4}", m.score);
                    }
                }
            }
        }

        // Calculate score based on:
        // 1. Number of perfect matches (score = 1.0)
        // 2. Coverage of required parameters
        // 3. Endpoint match confidence
        let perfect_matches = endpoint_specific_matches.iter()
            .filter(|(_, matches)| matches.iter().any(|m| m.score > 0.99))
            .count();

        let required_params = endpoint_config.parameters.iter()
            .filter(|p| p.required)
            .count();

        let matched_required = endpoint_specific_matches.iter()
            .filter(|(_, matches)| {
                matches.iter().any(|m| {
                    let param_name = extract_parameter_name(&m.key);
                    endpoint_config.parameters.iter()
                        .any(|p| p.name == param_name && p.required && m.score > 0.95)
                })
            })
            .count();

        let overall_score = 
            (perfect_matches as f32 / endpoint_config.parameters.len() as f32) * 0.4 +
            (matched_required as f32 / required_params as f32) * 0.4 +
            endpoint_match.score * 0.2
        ;

        debug!("\nDetailed scores for {}:", endpoint_id);
        debug!("- Perfect matches: {}/{}", perfect_matches, endpoint_config.parameters.len());
        debug!("- Required matches: {}/{}", matched_required, required_params);
        debug!("- Endpoint confidence: {:.4}", endpoint_match.score);
        debug!("- Overall score: {:.4}", overall_score);

        if overall_score > 0.6 {  // Lowered threshold as we're stricter now
            endpoint_matches.push(EndpointMatches {
                endpoint_id: endpoint_id.clone(),
                endpoint_confidence: endpoint_match.score,
                field_matches: endpoint_specific_matches.into_iter()
                    .flat_map(|(_, matches)| matches)
                    .map(|m| FieldMatch {
                        parameter_name: extract_parameter_name(&m.key),
                        endpoint_id: endpoint_id.clone(),
                        confidence: m.score,
                    })
                    .collect(),
                overall_score,
            });
        }
    }

    endpoint_matches.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());
    Ok(endpoint_matches)
    }
}

// Helper function to extract parameter name from key
fn extract_parameter_name(key: &str) -> String {
    // Format is "param:endpoint_id:parameter_name"
    key.split(':').nth(2).unwrap_or("unknown").to_string()
}

fn extract_base_param_name(key: &str) -> String {
    // For "param:send_email:to_alt:recipient_email", returns "to"
    key.split(':')
        .nth(2)  // Get the base parameter name part
        .map(|s| s.split("_alt").next().unwrap_or(s))
        .unwrap_or("unknown")
        .to_string()
}
