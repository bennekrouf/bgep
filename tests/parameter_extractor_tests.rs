use anyhow::Result;
use bgep::ParameterExtractor;

pub async fn setup() -> Result<ParameterExtractor> {
    let mut extractor = EXTRACTOR.lock().unwrap();
    if extractor.is_none() {
        *extractor = Some(ParameterExtractor::new("config.yaml")?);
    }
    Ok(extractor.as_ref().unwrap().clone())
}

#[tokio::test]
async fn test_email_extraction() -> Result<()> {
    let extractor = setup().await?;

    let test_cases = vec![
        ("send mail to test@example.com", Some("test@example.com")),
        (
            "multiple emails test1@example.com and test2@example.com",
            Some("test1@example.com"), // Should get first email
        ),
        ("no email here", None),
    ];

    for (input, expected) in test_cases {
        let params = extractor.extract_parameters(input)?;
        let email = params
            .iter()
            .find(|(param_type, _)| *param_type == "email")
            .map(|(_, value)| value.as_str());

        assert_eq!(email, expected, "Failed on input: {}", input);
    }

    Ok(())
}

#[tokio::test]
async fn test_title_extraction() -> Result<()> {
    let extractor = setup().await?;
    let test_cases = vec![
        (
            "document with title 'Test Title' here",
            Some("Test Title"),
            "Failed to extract simple title",
        ),
        (
            "no title markers here",
            None,
            "Should not extract title when none exists",
        ),
        (
            "title 'Multiple Words Title' test",
            Some("Multiple Words Title"),
            "Failed to extract multi-word title",
        ),
    ];

    for (input, expected, error_message) in test_cases {
        let params = extractor.extract_parameters(input)?;
        let title = params
            .iter()
            .find(|(param_type, _)| *param_type == "title")
            .map(|(_, value)| value.as_str());

        assert_eq!(
            title, expected,
            "{} - Input: '{}', Got: '{:?}', Expected: '{:?}'",
            error_message, input, title, expected
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_body_extraction() -> Result<()> {
    let extractor = setup().await?;

    let test_cases = vec![
        ("message with bodi 'Hello World' here", Some("Hello World")),
        ("no body markers here", None),
        (
            "body 'Multi-line\nBody Content' test",
            Some("Multi-line\nBody Content"),
        ),
    ];

    for (input, expected) in test_cases {
        let params = extractor.extract_parameters(input)?;
        let body = params
            .iter()
            .find(|(param_type, _)| *param_type == "body")
            .map(|(_, value)| value.as_str());

        assert_eq!(body, expected, "Failed on input: {}", input);
    }

    Ok(())
}

#[tokio::test]
async fn test_attachment_extraction() -> Result<()> {
    let extractor = setup().await?;

    let test_cases = vec![
        ("documen  xxx attached", Some("xxx")),
        ("file document.pdf here", Some("document.pdf")),
        ("no attachment here", None),
    ];

    for (input, expected) in test_cases {
        let params = extractor.extract_parameters(input)?;
        let attachment = params
            .iter()
            .find(|(param_type, _)| *param_type == "attachment")
            .map(|(_, value)| value.as_str());

        assert_eq!(attachment, expected, "Failed on input: {}", input);
    }

    Ok(())
}

#[tokio::test]
async fn test_multiple_parameters() -> Result<()> {
    let extractor = setup().await?;

    let input =
        "send document xxx to test@example.com with title 'Test Doc' and body 'Hello World'";
    let params = extractor.extract_parameters(input)?;

    let expected = vec![
        ("email", "test@example.com"),
        ("title", "Test Doc"),
        ("body", "Hello World"),
        ("attachment", "xxx"),
    ];

    for (expected_type, expected_value) in expected {
        let found = params
            .iter()
            .find(|(param_type, _)| *param_type == expected_type)
            .map(|(_, value)| value.as_str());

        assert_eq!(
            Some(expected_value),
            found,
            "Failed to match {}",
            expected_type
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_multilingual() -> Result<()> {
    let extractor = setup().await?;

    let input = "envoie à Jean le document xxx dont le title 'French Doc' and body 'Bonjour' à l'adresse email test@example.fr";
    let params = extractor.extract_parameters(input)?;

    let expected = vec![
        ("email", "test@example.fr"),
        ("title", "French Doc"),
        ("body", "Bonjour"),
        ("attachment", "xxx"),
    ];

    for (expected_type, expected_value) in expected {
        let found = params
            .iter()
            .find(|(param_type, _)| *param_type == expected_type)
            .map(|(_, value)| value.as_str());

        assert_eq!(
            Some(expected_value),
            found,
            "Failed to match {}",
            expected_type
        );
    }

    Ok(())
}
