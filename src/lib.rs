mod config;
mod embeddings_store;
mod json_processor;
mod matcher;
mod schema;

use anyhow::Result;
use arrow_array::{Float32Array, StringArray};
use arrow_schema::Schema;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;
use lancedb::query::QueryBase;
use lancedb::{connect, Connection, Table};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use arrow_array::Array;
use json_processor::JsonProcessor;
use std::sync::Arc;

use crate::schema::EMBEDDINGS_SCHEMA;
use arrow_array::RecordBatchIterator;
use futures_util::TryStreamExt;
use lancedb::query::ExecutableQuery;
use tokenizers::Tokenizer;
use tracing::{debug, info, info_span};

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};

use arrow_array::{ArrayRef, FixedSizeListArray};
use arrow_schema::{DataType, Field};

impl ParameterExtractor {
    pub async fn process_json(&self, json_str: &str) -> Result<Vec<MatchResult>> {
        let json_value: Value = serde_json::from_str(json_str)?;
        let mut results = Vec::new();

        // Get flattened items
        let items = JsonProcessor::flatten_json(json_value);

        // Process each item
        for item in items {
            match item.value {
                Value::String(s) => {
                    let matches = self.match_string(&s).await?;
                    if !matches.is_empty() {
                        results.push(MatchResult {
                            field_path: item.path,
                            value: s,
                            matches,
                        });
                    }
                }
                Value::Number(n) => {
                    let s = n.to_string();
                    let matches = self.match_string(&s).await?;
                    if !matches.is_empty() {
                        results.push(MatchResult {
                            field_path: item.path,
                            value: s,
                            matches,
                        });
                    }
                }
                _ => {} // Skip other types
            }
        }

        Ok(results)
    }
}

struct SingleBatchReader {
    batch: Option<RecordBatch>,
    schema: SchemaRef,
}

impl SingleBatchReader {
    fn new(batch: RecordBatch) -> Self {
        let schema = batch.schema();
        Self {
            batch: Some(batch),
            schema,
        }
    }
}

impl Iterator for SingleBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batch.take().map(Ok)
    }
}

impl RecordBatchReader for SingleBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParameterConfig {
    name: String,
    prompts: Vec<String>,
    threshold: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemConfig {
    parameters: Vec<ParameterConfig>,
    model_id: String,
    default_threshold: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MatchResult {
    pub field_path: String,
    pub value: String,
    pub matches: Vec<Match>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Match {
    pub parameter_type: String,
    pub confidence: f32,
}

pub struct ParameterExtractor {
    model: Arc<BertModel>,
    tokenizer: Tokenizer,
    device: Device,
    config: SystemConfig,
    connection: Connection,
    embeddings_table: Table,
    embeddings_schema: Arc<Schema>,
}

impl ParameterExtractor {
    pub async fn new(config_path: &str, db_path: &str) -> Result<Self> {
        let span = info_span!("initialize_extractor").entered();

        // Initialize device
        let device = Device::cuda_if_available(0)?;
        info!("Using device: {:?}", device);

        // Load config
        let config: SystemConfig = serde_yaml::from_str(&std::fs::read_to_string(config_path)?)?;

        // Initialize LanceDB
        let connection = connect(db_path).execute().await?;
        let embeddings_table = match connection.open_table("embeddings").execute().await {
            Ok(table) => table,
            Err(_) => {
                // Create new table if it doesn't exist
                let empty_batch = RecordBatch::new_empty(Arc::new(EMBEDDINGS_SCHEMA.clone()));
                let batch_iterator = RecordBatchIterator::new(
                    vec![Ok(empty_batch)],
                    Arc::new(EMBEDDINGS_SCHEMA.clone()),
                );
                connection
                    .create_table("embeddings", Box::new(batch_iterator))
                    .execute()
                    .await?
            }
        };

        // Initialize model
        let api = Api::new()?;
        let model_id = "BAAI/bge-large-en";
        let repo = api.model(model_id.to_string());

        let model_path = repo.get("model.safetensors")?;
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        info!("Loading BGE-Large-EN model...");

        let bert_config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        let vb = VarBuilder::from_tensors(
            candle_core::safetensors::load(model_path, &device)?,
            DType::F32,
            &device,
        );

        let model = Arc::new(BertModel::load(vb, &bert_config)?);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Pre-compute and store prompt embeddings
        let extractor = Self {
            model,
            tokenizer,
            device,
            config,
            connection,
            embeddings_table,
            embeddings_schema: Arc::new(EMBEDDINGS_SCHEMA.clone()),
        };

        extractor.initialize_prompts().await?;

        span.exit();
        Ok(extractor)
    }

    async fn initialize_prompts(&self) -> Result<()> {
        for param in &self.config.parameters {
            for prompt in &param.prompts {
                debug!("Processing prompt: {}", prompt);

                let embedding = self.get_embeddings(prompt)?;
                let embedding_vec: Vec<f32> = embedding.to_vec1()?;
                debug!(
                    "Embedding vector - length: {}, first few values: {:?}",
                    embedding_vec.len(),
                    &embedding_vec.iter().take(5).collect::<Vec<_>>()
                );

                let key = format!("prompt:{}:{}", param.name, prompt);
                let key_array = StringArray::from(vec![key.clone()]);
                let value_array = StringArray::from(vec![prompt.clone()]);
                let parameter_type_array = StringArray::from(vec![param.name.clone()]);

                // Create embedding array correctly
                let embedding_values = Arc::new(Float32Array::from(embedding_vec)) as ArrayRef;
                let embedding_field = Field::new("item", DataType::Float32, true);
                let embedding_array = FixedSizeListArray::new(
                    Arc::new(embedding_field),
                    1024,
                    embedding_values,
                    None, // no null values
                );

                debug!("Array lengths before batch creation:");
                debug!("  key_array: {}", key_array.len());
                debug!("  value_array: {}", value_array.len());
                debug!("  embedding_array: {}", embedding_array.len());
                debug!("  parameter_type_array: {}", parameter_type_array.len());

                let batch = RecordBatch::try_new(
                    self.embeddings_schema.clone(),
                    vec![
                        Arc::new(key_array),
                        Arc::new(value_array),
                        Arc::new(embedding_array),
                        Arc::new(parameter_type_array),
                    ],
                )?;

                debug!(
                    "RecordBatch created successfully - num rows: {}",
                    batch.num_rows()
                );

                let reader = SingleBatchReader::new(batch);
                debug!("Adding batch to table...");

                self.embeddings_table
                    .add(Box::new(reader))
                    .execute()
                    .await?;

                debug!("Successfully added batch to table");
            }
        }
        Ok(())
    }

    async fn match_string(&self, value: &str) -> Result<Vec<Match>> {
        let span = info_span!("match_string", value_len = value.len()).entered();
        let mut matches = Vec::new();

        let value_embedding = self.get_embeddings(value)?;
        let value_vec: Vec<f32> = value_embedding.to_vec1()?;

        // Query similar vectors from LanceDB and collect results
        let query_stream = self
            .embeddings_table
            .vector_search(value_vec)?
            .limit(5)
            .execute()
            .await?;

        let batches = query_stream.try_collect::<Vec<_>>().await?;

        for batch in batches {
            // Get the column indices
            let score_idx = batch.schema().index_of("_distance")?; // LanceDB stores distances in _distance column
            let param_type_idx = batch.schema().index_of("parameter_type")?;

            // Get arrays from the batch
            let score_array = batch
                .column(score_idx)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Failed to downcast score array"))?;

            let param_type_array = batch
                .column(param_type_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow::anyhow!("Failed to downcast parameter_type array"))?;

            // Process each row in the batch
            for row_idx in 0..batch.num_rows() {
                let score = score_array.value(row_idx);
                let parameter_type = param_type_array.value(row_idx).to_string();

                // Find the corresponding parameter config
                if let Some(param) = self
                    .config
                    .parameters
                    .iter()
                    .find(|p| p.name == parameter_type)
                {
                    if score > param.threshold {
                        matches.push(Match {
                            parameter_type,
                            confidence: score,
                        });
                    }
                }
            }
        }

        span.exit();
        Ok(matches)
    }

    fn get_embeddings(&self, text: &str) -> Result<Tensor> {
        let span = info_span!("get_embeddings", text_len = text.len()).entered();
        let start = std::time::Instant::now();

        debug!("Getting embeddings for text: {}", text);
        let processed_text = format!("Represent this text for retrieval: {}", text);

        let tokens = self
            .tokenizer
            .encode(processed_text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        debug!("Token length: {}", tokens.get_ids().len());

        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?;
        let attention_mask = Tensor::new(tokens.get_attention_mask(), &self.device)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?;

        debug!("Input shapes:");
        debug!("  input_ids: {:?}", input_ids.shape());
        debug!("  attention_mask: {:?}", attention_mask.shape());
        debug!("  token_type_ids: {:?}", token_type_ids.shape());

        let embeddings = self.model.forward(
            &input_ids.unsqueeze(0)?,
            &attention_mask.unsqueeze(0)?,
            Some(&token_type_ids.unsqueeze(0)?),
        )?;

        debug!("Raw embeddings shape: {:?}", embeddings.shape());

        // Convert attention mask to f32 and reshape
        let attention_mask = attention_mask
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(2)?;

        debug!(
            "Attention mask shape after reshape: {:?}",
            attention_mask.shape()
        );

        let attention_mask = attention_mask.broadcast_as(embeddings.shape())?;
        debug!(
            "Attention mask shape after broadcast: {:?}",
            attention_mask.shape()
        );

        let mean = embeddings.mul(&attention_mask)?;
        debug!("Mean shape: {:?}", mean.shape());

        let sum = mean.sum(1)?;
        debug!("Sum shape: {:?}", sum.shape());

        let counts = attention_mask.sum(1)?;
        debug!("Counts shape: {:?}", counts.shape());

        let embedding = sum.broadcast_div(&counts)?;
        debug!(
            "Embedding shape before normalization: {:?}",
            embedding.shape()
        );

        // Normalize the embedding
        let norm = embedding.sqr()?.sum_all()?.sqrt()?;
        let normalized = embedding.broadcast_div(&norm)?;
        debug!("Final normalized embedding shape: {:?}", normalized.shape());

        let final_embedding = normalized.squeeze(0)?;
        debug!(
            "Final squeezed embedding shape: {:?}",
            final_embedding.shape()
        );

        debug!(elapsed = ?start.elapsed(), "Computed embeddings");
        span.exit();

        Ok(final_embedding)
    }
}
