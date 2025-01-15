use anyhow::Result;
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures_util::TryStreamExt;
use lancedb::query::ExecutableQuery;
use lancedb::query::QueryBase;
use lancedb::{connect, Table};
use std::sync::Arc;
use tracing::info;
// use futures_util::TryStreamExt;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub key: String,
    pub value: String,
    pub endpoint_id: String,
    pub score: f32,
}

pub struct EmbeddingsStore {
    table: Table,
    schema: Arc<Schema>,
}

impl EmbeddingsStore {
    pub async fn is_empty(&self) -> Result<bool> {
        let count = self.table.count_rows(None).await?;
        Ok(count == 0)
    }

    pub async fn new(db_path: &str) -> Result<Self> {
        let db_path = "data/mydb";
        let connection = connect(db_path).execute().await?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Box::new(Field::new("item", DataType::Float32, true)).into(),
                    1024, // BGE-large-en embedding size
                ),
                false,
            ),
            Field::new("endpoint_id", DataType::Utf8, false),
        ]));

        let table = match connection.open_table("embeddings").execute().await {
            Ok(table) => table,
            Err(_) => {
                info!("Creating new embeddings table...");
                let empty_batch = RecordBatch::new_empty(schema.clone());
                let batch_iterator =
                    RecordBatchIterator::new(vec![Ok(empty_batch)], schema.clone());
                connection
                    .create_table("embeddings", Box::new(batch_iterator))
                    .execute()
                    .await?
            }
        };

        Ok(Self { table, schema })
    }

    pub async fn store_embedding(
        &self,
        key: &str,
        text: &str,
        endpoint_id: &str,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let key_array = StringArray::from(vec![key.to_string()]);
        let value_array = StringArray::from(vec![text.to_string()]);
        let type_array = StringArray::from(vec![endpoint_id.to_string()]);

        let embedding_values = Arc::new(Float32Array::from(embedding)) as ArrayRef;
        let embedding_field = Field::new("item", DataType::Float32, true);
        let embedding_array =
            FixedSizeListArray::new(Arc::new(embedding_field), 1024, embedding_values, None);

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(key_array),
                Arc::new(value_array),
                Arc::new(embedding_array),
                Arc::new(type_array),
            ],
        )?;

        // Create iterator for the single batch
        let batch_iterator = RecordBatchIterator::new(vec![Ok(batch)], self.schema.clone());

        self.table.add(Box::new(batch_iterator)).execute().await?;

        Ok(())
    }

    pub async fn search_similar(
        &self,
        embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let results = self
            .table
            .vector_search(embedding)?
            .limit(limit)
            .execute()
            .await?;

        let mut search_results = Vec::new();
        let mut stream = results;
        while let Some(record) = stream.try_next().await? {
            // Get column indices
            let key_idx = record.schema().index_of("key")?;
            let value_idx = record.schema().index_of("value")?;
            let endpoint_id_idx = record.schema().index_of("endpoint_id")?;
            let distance_idx = record.schema().index_of("_distance")?; // LanceDB stores scores as distances

            // Get arrays from columns
            let key_array = record
                .column(key_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow::anyhow!("Failed to downcast key array"))?;

            let value_array = record
                .column(value_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow::anyhow!("Failed to downcast value array"))?;

            let endpoint_id_array = record
                .column(endpoint_id_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow::anyhow!("Failed to downcast endpoint_id array"))?;

            let distance_array = record
                .column(distance_idx)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Failed to downcast distance array"))?;

            // Add each row to results
            for row_idx in 0..record.num_rows() {
                search_results.push(SearchResult {
                    key: key_array.value(row_idx).to_string(),
                    value: value_array.value(row_idx).to_string(),
                    endpoint_id: endpoint_id_array.value(row_idx).to_string(),
                    score: 1.0 - distance_array.value(row_idx), // Convert distance to similarity score
                });
            }
        }

        Ok(search_results)
    }
}
