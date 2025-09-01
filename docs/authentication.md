# HuggingFace Authentication Guide

This document explains how to set up HuggingFace authentication for accessing gated models like Llama 3.

## Why Authentication is Needed

Some models on HuggingFace Hub are gated and require authentication:
- Meta Llama models (Llama 2, Llama 3, CodeLlama)
- Many research models
- Private repositories

## Setup Methods

### Method 1: Environment Variable (Recommended)

Set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

Or on Windows:
```cmd
set HF_TOKEN=your_huggingface_token_here
```

### Method 2: Configuration File

Add the token to your configuration file:

```yaml
model:
  base_model_id: "meta-llama/Meta-Llama-3-8B"
  hf_token: "your_huggingface_token_here"
  use_auth_token: true
```

### Method 3: HuggingFace CLI Login

Use the HuggingFace CLI to log in once:

```bash
pip install huggingface_hub
huggingface-cli login
```

## Getting Your Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token or copy an existing one
3. Make sure it has the necessary permissions:
   - `read` for downloading models
   - `write` if you plan to upload models

## Configuration Options

In your YAML config file:

```yaml
model:
  base_model_id: "meta-llama/Meta-Llama-3-8B"
  use_auth_token: true    # Enable/disable authentication
  hf_token: null          # Token (optional if using env var)
```

## Testing Authentication

Run the test script to verify your setup:

```bash
python test_auth.py
```

This will:
- Check if authentication token is available
- Test authentication setup
- Verify access to gated models
- Provide troubleshooting guidance

## Error Messages

### "Model may require authentication"
- The model you're trying to use is gated
- Set up authentication using one of the methods above

### "No HuggingFace token found"
- No token is configured
- Set HF_TOKEN environment variable or add hf_token to config

### "Failed to login to HuggingFace Hub"
- Your token may be invalid or expired
- Check your token in HuggingFace settings
- Ensure the token has the correct permissions

## Troubleshooting

1. **Token not working?**
   - Verify the token is correct
   - Check token permissions
   - Try regenerating the token

2. **Still getting access errors?**
   - Ensure you've requested access to the gated model
   - Some models require manual approval from the model authors

3. **Environment variable not working?**
   - Restart your terminal/IDE after setting the variable
   - Use `echo $HF_TOKEN` (Linux/Mac) or `echo %HF_TOKEN%` (Windows) to verify

## Best Practices

1. **Use environment variables** for security (don't commit tokens to code)
2. **Set minimal permissions** on your tokens
3. **Regenerate tokens** periodically for security
4. **Test authentication** before running long training jobs

## Security Notes

- Never commit tokens to version control
- Use environment variables or secure configuration management
- Regularly rotate your tokens
- Use read-only tokens when possible