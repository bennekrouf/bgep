use crate::config::Config;
use std::collections::HashSet;
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
pub struct Match {
    pub field: String,
    pub text: String,
    pub similar_fields: Vec<SearchResult>,
}

#[derive(Debug)]
pub struct IntentMatch {
    pub endpoint_id: String,
    pub confidence: f32,
}

#[derive(Debug)]
pub struct ParameterMatch {
    pub field: String,
    pub text: String,
    pub matches: Vec<FieldMatch>,
}

#[derive(Debug)]
pub struct FieldMatch {
    pub parameter_name: String,
    pub endpoint_id: String,
    pub confidence: f32,
}

#[derive(Debug)]
pub struct CompleteMatch {
    pub endpoint: IntentMatch,
    pub parameters: Vec<ParameterMatch>,
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
        info!("Initializing embeddings for config...");
        for endpoint in &self.config.endpoints {
            // Store parameter field name embeddings with full context
            for param in &endpoint.parameters {
                let param_context = format!(
                    "Parameter '{}' for {} operation: {}",
                    param.name,
                    endpoint.text,
                    param.description
                );
                debug!("Computing embedding for parameter context: {}", param_context);
                let param_embedding = self.compute_embedding(&param_context)?;
                self.embeddings_store
                    .store_embedding(
                        &format!("param:{}:{}", endpoint.id, param.name),
                        &param_context,
                        &endpoint.id,
                        param_embedding,
                    )
                    .await?;

                // Store alternatives with context
                if let Some(alternatives) = &param.alternatives {
                    for alt in alternatives {
                        let alt_context = format!(
                            "Alternative field name '{}' for parameter '{}' in {} operation: {}",
                            alt,
                            param.name,
                            endpoint.text,
                            param.description
                        );
                        debug!("Computing embedding for alternative context: {}", alt_context);
                        let alt_embedding = self.compute_embedding(&alt_context)?;
                        self.embeddings_store
                            .store_embedding(
                                &format!("param:{}:{}_alt:{}", endpoint.id, param.name, alt),
                                &alt_context,
                                &endpoint.id,
                                alt_embedding,
                            )
                            .await?;
                    }
                }
            }
        }
        info!("Finished initializing embeddings");
        Ok(())
    }

    pub async fn match_json_two_phase(&self, json_str: &str) -> Result<Vec<Match>> {
        let json: Value = serde_json::from_str(json_str)?;
        let mut matches = Vec::new();

        // Phase 1: Match intent only with endpoint text and descriptions
        let request = json["request"].as_str()
            .ok_or_else(|| anyhow::anyhow!("No request field found"))?;
        debug!("\nAnalyzing request: '{}'", request);
        let request_embedding = self.compute_embedding(request)?;

        // Get all matches first for debugging
        let all_matches = self.embeddings_store
            .search_similar(request_embedding, 10)
            .await?;

        debug!("\nAll matches for request:");
        for m in &all_matches {
            debug!("- {} (score: {:.4})", m.key, m.score);
        }

        // Get unique endpoint matches
        let unique_endpoints: Vec<_> = all_matches.into_iter()
            .filter(|m| {
                let is_endpoint = m.key.starts_with("endpoint:");
                let is_pure_endpoint = !m.key.ends_with("_desc");
                debug!("Matching '{}' against endpoint '{}': score={:.4}", 
                    request, m.key, m.score);
                debug!("  is_endpoint={}, is_pure_endpoint={}", 
                    is_endpoint, is_pure_endpoint);
                is_endpoint && is_pure_endpoint
            })
            .collect();

        // Remove duplicates by key
        let mut seen_keys = std::collections::HashSet::new();
        let potential_endpoints: Vec<_> = unique_endpoints.into_iter()
            .filter(|result| seen_keys.insert(result.key.clone()))
            .collect();

        debug!("\nFinal endpoint matches:");
        for ep in &potential_endpoints {
            debug!("- {} (score: {:.4})", ep.key, ep.score);
        }

        // Phase 2: Match field names only, not their values
        if let Value::Object(map) = json {
            for field in map.keys() {
                if field != "request" {
                    debug!("Matching field name: {}", field);

                    let field_context = format!(
                        "Field name '{}' for potential parameter matching",
                        field
                    );

                    let field_embedding = self.compute_embedding(&field_context)?;
                    let similar = self
                        .embeddings_store
                        .search_similar(field_embedding, 5)
                        .await?
                        .into_iter()
                        .filter(|m| m.key.starts_with("param:")) // Only consider parameter matches
                        .collect::<Vec<_>>();

                    matches.push(Match {
                        field: field.clone(),
                        text: field.clone(), // Use field name instead of value
                        similar_fields: similar,
                    });
                }
            }
        }

        Ok(matches)
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

    pub async fn match_json(&self, json_str: &str) -> Result<Vec<Match>> {
        let json: Value = serde_json::from_str(json_str)?;
        let mut matches = Vec::new();

        // Match each string field in the JSON
        if let Value::Object(map) = json {
            for (field, value) in map {
                if let Value::String(text) = value {
                    debug!("Computing embedding for field {}: {}", field, text);

                    // Create a contextual embedding using the field name
                    let contextual_text = format!("{} field with value: {}", field, text);
                    let embedding = self.compute_embedding(&contextual_text)?;
                    let similar = self.embeddings_store.search_similar(embedding, 5).await?;

                    matches.push(Match {
                        field,
                        text: text.to_string(),
                        similar_fields: similar,
                    });
                }
            }
        }

        Ok(matches)
    }

    // Phase 1: Match the intent
    async fn match_intent(&self, request: &str) -> Result<IntentMatch> {
        debug!("Matching intent for request: {}", request);
        let embedding = self.compute_embedding(request)?;

        // Search for endpoint matches (using endpoint text and description)
        let similar = self.embeddings_store.search_similar(embedding, 3).await?;

        // Get the best match
        let best_match = similar
            .into_iter()
            .find(|result| result.key.starts_with("endpoint:"))
            .ok_or_else(|| anyhow::anyhow!("No matching endpoint found"))?;

        Ok(IntentMatch {
            endpoint_id: best_match.endpoint_id,
            confidence: best_match.score,
        })
    }

    // Phase 2: Match parameters for the identified endpoint
    async fn match_parameters(
        &self,
        intent: &IntentMatch,
        json: &Value,
    ) -> Result<Vec<ParameterMatch>> {
        let mut matches = Vec::new();

        // Get the endpoint configuration
        let endpoint = self
            .config
            .endpoints
            .iter()
            .find(|e| e.id == intent.endpoint_id)
            .ok_or_else(|| anyhow::anyhow!("Endpoint not found"))?;

        // Process each field in the JSON except 'request'
        if let Value::Object(map) = json {
            for (field, value) in map {
                if field != "request" {
                    if let Value::String(text) = value {
                        // Create contextual embedding using the endpoint
                        let context = format!(
                            "For {} operation, field {} with value: {}",
                            endpoint.description, field, text
                        );
                        let embedding = self.compute_embedding(&context)?;

                        // Search for parameter matches within this endpoint
                        let similar = self.embeddings_store.search_similar(embedding, 3).await?;

                        let field_matches: Vec<FieldMatch> = similar
                            .clone()
                            .into_iter()
                            .map(|match_result| FieldMatch {
                                parameter_name: extract_parameter_name(&match_result.key),
                                endpoint_id: match_result.endpoint_id,
                                confidence: match_result.score,
                            })
                            .collect();

                        // Find best parameter match
                        if let Some(best_match) = similar.into_iter().find(|result| {
                            result
                                .key
                                .starts_with(&format!("param:{}:", intent.endpoint_id))
                        }) {
                            matches.push(ParameterMatch {
                                field: field.clone(),
                                text: text.clone(),
                                matches: field_matches,
                            });
                        }
                    }
                }
            }
        }

        Ok(matches)
    }

    pub async fn match_json_holistic(&self, json_str: &str) -> Result<Vec<EndpointMatches>> {
        let json: Value = serde_json::from_str(json_str)?;
        let mut endpoint_matches = Vec::new();

        // Phase 1: Get endpoint matches from request
        let request = json["request"].as_str()
            .ok_or_else(|| anyhow::anyhow!("No request field found"))?;
        debug!("\nAnalyzing request: '{}'", request);
        let request_embedding = self.compute_embedding(request)?;

        // Get all matches first for debugging
        let all_matches = self.embeddings_store
            .search_similar(request_embedding, 10)
            .await?;

        debug!("\nAll matches for request:");
        for m in &all_matches {
            debug!("- {} (score: {:.4})", m.key, m.score);
        }

        // First try to get pure endpoint matches
        let mut potential_endpoints: Vec<_> = all_matches.clone().into_iter()
            .filter(|m| {
                let is_endpoint = m.key.starts_with("endpoint:");
                let is_pure_endpoint = !m.key.ends_with("_desc");
                debug!("Matching '{}' against endpoint '{}': score={:.4}", 
                    request, m.key, m.score);
                debug!("  is_endpoint={}, is_pure_endpoint={}", 
                    is_endpoint, is_pure_endpoint);
                        is_endpoint && is_pure_endpoint
                    })
                    .collect();

        // If no pure endpoints found, fall back to all endpoint matches
        if potential_endpoints.is_empty() {
            debug!("No pure endpoint matches found, falling back to all endpoint matches");
            potential_endpoints = all_matches.into_iter()
                .filter(|m| m.key.starts_with("endpoint:"))
                .collect();
        }

        debug!("\nFinal endpoint matches:");
        for ep in &potential_endpoints {
            debug!("- {} (score: {:.4})", ep.key, ep.score);
        }

        let potential_endpoints = potential_endpoints.into_iter()
            .filter(|m| m.key.starts_with("endpoint:"))
            .collect::<Vec<_>>();

        // Phase 2: For each potential endpoint, evaluate field matches
        for endpoint_match in potential_endpoints {
        let endpoint_id = &endpoint_match.endpoint_id;


        // Get the endpoint configuration
        let endpoint_config = self.config.endpoints.iter()
            .find(|e| &e.id == endpoint_id)
            .ok_or_else(|| anyhow::anyhow!("Endpoint not found"))?;

        let mut endpoint_specific_matches = Vec::new();

        // Get all field matches for this endpoint first
        if let Value::Object(map) = &json {
            for field_name in map.keys() {
                if field_name != "request" {
                    debug!("\nAnalyzing field '{}' for endpoint '{}':", field_name, endpoint_id);
                    let field_embedding = self.compute_embedding(field_name)?;
                    let matches = self.embeddings_store
                        .search_similar(field_embedding, 3)
                        .await?;

                    debug!("Field '{}' matches:", field_name);
                    for m in &matches {
                        debug!("  - {} (score: {:.4})", m.key, m.score);
                    }

                    let endpoint_matches: Vec<_> = matches.clone().into_iter()
                        .filter(|m| m.key.starts_with(&format!("param:{}:", endpoint_id)))
                        .collect();

                    if !endpoint_matches.is_empty() {
                        debug!("Matched parameters for field '{}':", field_name);
                        for m in &endpoint_matches {
                            debug!("  - {} (score: {:.4})", m.key, m.score);
                        }
                        endpoint_specific_matches.push((field_name, endpoint_matches));
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
