use arrow_schema::{DataType, Field, Schema};
use lazy_static::lazy_static;

lazy_static! {
    pub static ref EMBEDDINGS_SCHEMA: Schema = Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, false),
        Field::new("embedding", DataType::FixedSizeList(
            *Box::new(Field::new("item", DataType::Float32, true).into()),
            1024  // BGE-large-en embedding size
        ), false),
        Field::new("parameter_type", DataType::Utf8, false),
    ]);
}
