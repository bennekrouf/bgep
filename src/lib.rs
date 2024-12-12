use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{ops::sigmoid, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub struct ParameterExtractor {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl ParameterExtractor {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;
        let api = Api::new()?;
        let model_id = "katanemo/bge-large-en-v1.5";
        let _revision = "main";
        let repo = api.model(model_id.to_string());

        let model_path = repo.get("model.safetensors")?;
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        let vb = VarBuilder::from_tensors(
            candle_core::safetensors::load(model_path, &device)?,
            DType::F32,
            &device,
        );
        let model = BertModel::load(vb, &config)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn extract_parameters(&self, text: &str) -> Result<Vec<(String, String)>> {
        let text_embedding = self.get_embeddings(text)?;
        let mut parameters = Vec::new();

        let parameter_patterns = vec![
            ("email", "an email address"),
            ("title", "a document title"),
            ("body", "a message body"),
            ("attachment", "a file or document name"),
        ];

        for (param_type, description) in parameter_patterns {
            let prompt = format!("Find {} in the text", description);
            let pattern_embedding = self.get_embeddings(&prompt)?;

            let pattern_embedding = pattern_embedding.transpose(0, 1)?;
            let similarity = text_embedding.matmul(&pattern_embedding)?;
            let score = sigmoid(&similarity)?.get(0)?.get(0)?.to_scalar::<f32>()?;

            if score > 0.5 {
                let value = self.extract_value(text, param_type)?;
                if !value.is_empty() {
                    parameters.push((param_type.to_string(), value));
                }
            }
        }

        Ok(parameters)
    }

    fn get_embeddings(&self, text: &str) -> Result<Tensor> {
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
    }

    fn extract_value(&self, text: &str, param_type: &str) -> Result<String> {
        match param_type {
            "email" => {
                if let Some(email) = text.split_whitespace().find(|word| word.contains('@')) {
                    Ok(email.to_string())
                } else {
                    Ok(String::new())
                }
            }
            "title" => {
                if let Some(title_start) = text.find("title '") {
                    let start_idx = title_start + "title '".len();
                    if let Some(end_quote) = text[start_idx..].find('\'') {
                        Ok(text[start_idx..start_idx + end_quote].to_string())
                    } else {
                        Ok(String::new())
                    }
                } else {
                    Ok(String::new())
                }
            }
            "body" => {
                if let Some(body_start) = text.find("body '") {
                    let start_idx = body_start + "body '".len();
                    if let Some(end_quote) = text[start_idx..].find('\'') {
                        Ok(text[start_idx..start_idx + end_quote].to_string())
                    } else {
                        Ok(String::new())
                    }
                } else {
                    Ok(String::new())
                }
            }
            "attachment" => {
                if let Some(doc) = text
                    .split_whitespace()
                    .find(|word| word.ends_with(".pdf") || word.ends_with(".doc") || *word == "xxx")
                {
                    Ok(doc.to_string())
                } else {
                    Ok(String::new())
                }
            }
            _ => Ok(String::new()),
        }
    }
}
