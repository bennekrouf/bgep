use anyhow::Result;
use candle_core::{DType, Device, Tensor, utils::cuda_is_available};
use candle_nn::{ops::sigmoid, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use lazy_static::lazy_static;
use std::sync::Arc;
use tracing::{info, warn, info_span, debug};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParameterConfig {
    name: String,
    prompts: Vec<String>,
    threshold: f32,
    context_window: usize,  // How many words to look around markers
    markers: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemConfig {
    parameters: Vec<ParameterConfig>,
    model_id: String,
    default_threshold: f32,
}

lazy_static! {
    pub static ref EXTRACTOR: Mutex<Option<ParameterExtractor>> = Mutex::new(None);
}

async fn setup() -> Result<ParameterExtractor> {
    let mut extractor = EXTRACTOR.lock().unwrap();
    if extractor.is_none() {
        *extractor = Some(ParameterExtractor::new("config.yaml")?);
    }
    Ok(extractor.as_ref().unwrap().clone())
}

#[derive(Clone)]
pub struct ParameterExtractor {
    model: Arc<BertModel>,
    tokenizer: Tokenizer,
    device: Device,
    config: SystemConfig,
    prompt_embeddings: HashMap<String, Tensor>,  // Cache for prompt embeddings
}

impl ParameterExtractor {
    pub fn new(config_path: &str) -> Result<Self> {

        // Try CUDA first, fallback to CPU
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("Device in use: {}", check_device(&device));

        // Also log available devices
        let dev = cuda_is_available(); 
        if dev {
            info!("CUDA available: {} device(s)", dev);
        } else {
            info!("CUDA not available");
        }

        let span = info_span!("initialize_extractor").entered();
        // let device = Device::Cpu;
        let device = Device::cuda_if_available(0)?;
        let config: SystemConfig = serde_yaml::from_str(&std::fs::read_to_string(config_path)?)?;

        let api = Api::new()?;
        let model_id = "katanemo/bge-large-en-v1.5";
        let _revision = "main";
        let repo = api.model(model_id.to_string());

        let model_path = repo.get("model.safetensors")?;
        let bert_config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        let bert_config: Config = serde_json::from_str(&std::fs::read_to_string(bert_config_path)?)?;

        let vb = VarBuilder::from_tensors(
            candle_core::safetensors::load(model_path, &device)?,
            DType::F32,
            &device,
        );

        info!("Loading model and tokenizer...");
        let model = Arc::new(BertModel::load(vb, &bert_config)?);

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        info!("Computing prompt embeddings...");
        let start = std::time::Instant::now();

        let mut prompt_embeddings = HashMap::new();
        for param in &config.parameters {
            for prompt in &param.prompts {
                let embedding = compute_embedding(&model, &tokenizer, &device, prompt)?;
                prompt_embeddings.insert(prompt.clone(), embedding);
            }
        }

        info!("Computed {} prompt embeddings in {:?}", prompt_embeddings.len(), start.elapsed());
        span.exit();

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            prompt_embeddings,
        })
    }

    pub fn extract_parameters(&self, text: &str) -> Result<HashMap<String, String>> {
        let span = info_span!("extract_parameters", text_len = text.len()).entered();
        let start = std::time::Instant::now();

        let mut results = HashMap::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for param_config in &self.config.parameters {
            let param_start = std::time::Instant::now();
            // Try marker-based extraction first without semantic validation
            if let Some(value) = self.extract_by_markers(text, &words, param_config) {
                info!(
                    parameter = param_config.name.as_str(),
                    elapsed = ?param_start.elapsed(),
                    "Found parameter using markers"
                );
                results.insert(param_config.name.clone(), value);
                continue;
            }

            // Only use semantic extraction for unmatched parameters
            if !results.contains_key(&param_config.name) {
                if let Some(value) = self.extract_by_semantics(text, &words, param_config)? {
                    info!(
                        parameter = param_config.name.as_str(),
                        elapsed = ?param_start.elapsed(),
                        "Found parameter using semantics"
                    );
                    results.insert(param_config.name.clone(), value);
                }
            }
        }

        info!(total_elapsed = ?start.elapsed(), "Parameter extraction complete");
        span.exit();
        Ok(results)
    }

    fn extract_by_markers(&self, text: &str, words: &[&str], config: &ParameterConfig) -> Option<String> {
        // Special handling for attachments and emails - pure pattern matching
        match config.name.as_str() {
            "attachment" => {
                words.iter()
                    .find(|&&word| {
                        word.ends_with(".pdf") || 
                        word.ends_with(".doc") || 
                        word.ends_with(".txt") || 
                        word == "xxx"
                    })
                    .map(|&word| word.to_string())
            }
            "email" => {
                words.iter()
                    .find(|&&word| word.contains('@'))
                    .map(|&word| word.to_string())
            }
            _ => {
                // For other types, look for markers and quotes
                for (idx, word) in words.iter().enumerate() {
                    if config.markers.iter().any(|marker| 
                        word.to_lowercase().contains(&marker.to_lowercase())
                    ) {
                        // Try quotes first
                        if let Some(quote_start) = text[text.find(word)?..].find('\'') {
                            if let Some(quote_end) = text[text.find(word)? + quote_start + 1..].find('\'') {
                                return Some(text[text.find(word)? + quote_start + 1..
                                            text.find(word)? + quote_start + quote_end + 1].to_string());
                            }
                        }

                        // If no quotes, only do semantic validation for title and body
                        if config.name == "title" || config.name == "body" {
                            let start_idx = (idx + 1).min(words.len());
                            let end_idx = (idx + 1 + config.context_window).min(words.len());

                            if start_idx < end_idx {
                                let extracted = words[start_idx..end_idx].join(" ");
                                if let Ok(score) = self.validate_extraction(&extracted, config) {
                                    if score > config.threshold {
                                        return Some(extracted);
                                    }
                                }
                            }
                        }
                    }
                }
                None
            }
        }
    }

    fn extract_by_semantics(&self, text: &str, words: &[&str], config: &ParameterConfig) -> Result<Option<String>> {
        let span = info_span!("semantic_extraction", parameter = config.name.as_str()).entered();
        let start = std::time::Instant::now();

        let result = {
            let mut best_score = 0.0;
            let mut best_extraction = None;

            // Create sliding windows of different sizes
            for window_size in 2..=config.context_window {
                for window in words.windows(window_size) {
                    let chunk = window.join(" ");
                    let score = self.validate_extraction(&chunk, config)?;

                    if score > best_score {
                        best_score = score;
                        best_extraction = Some(chunk.to_string());
                    }
                }
            }

            if best_score > config.threshold {
                Ok(best_extraction)
            } else {
                Ok(None)
            }
        };

        info!(elapsed = ?start.elapsed(), "Semantic extraction complete");
        span.exit();
        result
    }

    fn validate_extraction(&self, text: &str, config: &ParameterConfig) -> Result<f32> {
        let text_embedding = self.get_embeddings(text)?;
        let mut max_score:f32 = 0.0;

        for prompt in &config.prompts {
            if let Some(prompt_embedding) = self.prompt_embeddings.get(prompt) {
                let score = self.compute_similarity(&text_embedding, prompt_embedding)?;
                max_score = max_score.max(score);
            }
        }
        Ok(max_score)
    }

    fn get_embeddings(&self, text: &str) -> Result<Tensor> {
        let span = info_span!("get_embeddings", text_len = text.len()).entered();
        let start = std::time::Instant::now();

        let result = {
            let tokens = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

            let input_ids = Tensor::new(tokens.get_ids(), &self.device)?;
            let attention_mask = Tensor::new(tokens.get_attention_mask(), &self.device)?;
            let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?;

            let embeddings = self.model.forward(
                &input_ids.unsqueeze(0)?,
                &attention_mask.unsqueeze(0)?,
                Some(&token_type_ids.unsqueeze(0)?),
            )?;

            let cls_embedding = embeddings.get(0)?.get(0)?.clone();
            let norm = cls_embedding.sqr()?.sum_all()?.sqrt()?;
            let normalized = cls_embedding.broadcast_div(&norm)?;
            Ok(normalized.unsqueeze(0)?)
        };

        info!(elapsed = ?start.elapsed(), "Computed embeddings");
        span.exit();
        result
    }

    fn compute_similarity(&self, embedding1: &Tensor, embedding2: &Tensor) -> Result<f32> {
        let similarity = embedding1.matmul(&embedding2.transpose(0, 1)?)?;
        let score = sigmoid(&similarity)?.get(0)?.get(0)?.to_scalar::<f32>()?;
        Ok(score)
    }
}


fn compute_embedding(model: &BertModel, tokenizer: &Tokenizer, device: &Device, text: &str) -> Result<Tensor> {
    let total_start = std::time::Instant::now();

    // Tokenization timing
    let tokenize_start = std::time::Instant::now();
    let tokens = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    info!(elapsed = ?tokenize_start.elapsed(), "Tokenization completed");

    // Tensor creation timing
    let tensor_start = std::time::Instant::now();
    let input_ids = Tensor::new(tokens.get_ids(), device)?;
    let attention_mask = Tensor::new(tokens.get_attention_mask(), device)?;
    let token_type_ids = Tensor::new(tokens.get_type_ids(), device)?;
    info!(elapsed = ?tensor_start.elapsed(), "Tensor creation completed");

    // Model forward pass timing
    let forward_start = std::time::Instant::now();
    let embeddings = model.forward(
        &input_ids.unsqueeze(0)?,
        &attention_mask.unsqueeze(0)?,
        Some(&token_type_ids.unsqueeze(0)?),
    )?;
    info!(elapsed = ?forward_start.elapsed(), "Model forward pass completed");

    // Post-processing timing
    let postproc_start = std::time::Instant::now();
    let cls_embedding = embeddings.get(0)?.get(0)?.clone();
    let norm = cls_embedding.sqr()?.sum_all()?.sqrt()?;
    let normalized = cls_embedding.broadcast_div(&norm)?;
    let result = normalized.unsqueeze(0)?;
    info!(elapsed = ?postproc_start.elapsed(), "Post-processing completed");

    info!(
        total_elapsed = ?total_start.elapsed(),
        text_len = text.len(),
        "Complete embedding computation"
    );

    Ok(result)
}

fn check_device(device: &Device) -> String {
    match device {
        Device::Cpu => "Running on CPU".to_string(),
        Device::Cuda(n) => format!("Running on CUDA GPU {:?}", n),
        Device::Metal(_) => "Running on Metal".to_string(),
        // _ => "Running on unknown device".to_string(),
    }
}
