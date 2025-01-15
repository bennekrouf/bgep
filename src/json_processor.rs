use serde_json::Value;
use std::collections::VecDeque;

#[derive(Debug)]
pub struct ProcessingItem {
    pub value: Value,
    pub path: String,
}

pub struct JsonProcessor;

impl JsonProcessor {
    pub fn flatten_json(json_value: Value) -> VecDeque<ProcessingItem> {
        let mut items = VecDeque::new();
        let mut stack = VecDeque::new();

        // Push initial value
        stack.push_back(ProcessingItem {
            value: json_value,
            path: String::new(),
        });

        // Process all items in stack
        while let Some(current) = stack.pop_front() {
            match current.value {
                Value::Object(map) => {
                    for (key, val) in map {
                        let new_path = if current.path.is_empty() {
                            key
                        } else {
                            format!("{}.{}", current.path, key)
                        };

                        match val {
                            Value::Object(_) | Value::Array(_) => {
                                stack.push_back(ProcessingItem {
                                    value: val,
                                    path: new_path,
                                });
                            }
                            _ => {
                                items.push_back(ProcessingItem {
                                    value: val,
                                    path: new_path,
                                });
                            }
                        }
                    }
                }
                Value::Array(arr) => {
                    for (idx, val) in arr.into_iter().enumerate() {
                        let new_path = format!("{}[{}]", current.path, idx);

                        match val {
                            Value::Object(_) | Value::Array(_) => {
                                stack.push_back(ProcessingItem {
                                    value: val,
                                    path: new_path,
                                });
                            }
                            _ => {
                                items.push_back(ProcessingItem {
                                    value: val,
                                    path: new_path,
                                });
                            }
                        }
                    }
                }
                _ => {
                    items.push_back(current);
                }
            }
        }

        items
    }
}
