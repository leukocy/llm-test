# Security Guide

## Overview

This document describes the security features of the LLM Test Platform and how to use them securely.

## Security Features

### 1. Code Execution Safety

The platform uses restricted execution environments for code evaluation:

- **What's protected**: HumanEval code execution, math expression evaluation
- **How it works**: Uses AST validation and restricted builtins
- **What's blocked**: File I/O, network operations, imports, arbitrary code execution

### 2. SSRF Protection

URL validation prevents Server-Side Request Forgery:

- **Default**: Blocks private IPs, loopback addresses, and non-HTTP protocols
- **Development**: Can be disabled with `allow_private_urls=True`
- **Production**: Always use `allow_private_urls=False`

### 3. Path Traversal Protection

File access is restricted to the datasets directory:

- **What's protected**: Dataset loader, file uploads
- **How it works**: Validates paths stay within allowed directories

### 4. Rate Limiting

API calls are rate-limited to prevent abuse:

- **Default**: 10 requests/second with burst of 20
- **Configurable**: Modify `get_rate_limiter()` call

## Best Practices

### For Development

1. Use environment variables for all credentials
2. Run with `LLM_TEST_DEV=true` only when testing internal servers
3. Keep `.env` file in `.gitignore`
4. Never commit `.env` or hardcoded keys

### For Production

1. Set `LLM_TEST_USERNAME` and `LLM_TEST_PASSWORD_HASH` for authentication
2. Use HTTPS only (set `require_https=True` in URL validation)
3. Disable development mode
4. Use a secrets manager (e.g., HashiCorp Vault, AWS Secrets Manager)
5. Review and audit the code regularly
6. Keep dependencies updated

## Environment Variables

### Required for Production

```bash
# API Keys (never hardcode these!)
export ALIYUN_API_KEY=your_key_here
export GEMINI_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export DEEPSEEK_API_KEY=your_key_here
```

### Optional Authentication

```bash
# Enable authentication
export LLM_TEST_USERNAME=admin
export LLM_TEST_PASSWORD_HASH=$(echo -n "your_password" | sha256sum)
```

### Development Mode

```bash
# Enable development mode (allows internal servers)
export LLM_TEST_DEV=true
```

### Configuration

```bash
# Maximum workers for concurrent requests
export MAX_WORKERS=100

# API timeout in seconds
export API_TIMEOUT=120
```

## Security Auditing

Run security tests:

```bash
# Run all security tests
pytest tests/test_security.py -v

# Check for hardcoded secrets
grep -r "sk-" --include="*.py" . | grep -v ".env"
grep -r "AIzaSy" --include="*.py" . | grep -v ".env"

# Check for unsafe eval/exec
grep -r "exec(" --include="*.py" . | grep -v "test_" | grep -v "safe_executor"
grep -r "eval(" --include="*.py" . | grep -v "test_" | grep -v "safe_executor"
```

## Vulnerability Reporting

If you find a security vulnerability, please report it privately to the maintainers.

Do NOT:
- Publicly disclose the vulnerability
- Exploit the vulnerability
- Share the vulnerability with others

Do:
- Report privately through responsible disclosure
- Provide details on how to reproduce
- Suggest a fix if possible

## Security Checklist

### Before Deployment

- [ ] All API keys in environment variables (not hardcoded)
- [ ] HTTPS enforced for all API calls
- [ ] Authentication enabled (if needed)
- [ ] Development mode disabled
- [ ] Security tests passing
- [ ] No hardcoded secrets in code
- [ ] Dependencies updated and scanned
- [ ] Rate limiting configured
- [ ] Log sanitization enabled

### Regular Maintenance

- [ ] Update dependencies monthly
- [ ] Review access logs weekly
- [ ] Rotate API keys quarterly
- [ ] Audit code for security issues
- [ ] Run security scans

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [RestrictedPython Documentation](https://restrictedpython.readthedocs.io/)

## Changelog

### Version 1.0.0 (Current)

- Added safe code execution with AST validation
- Added SSRF protection with URL validation
- Added path traversal protection
- Added secure API key management
- Added rate limiting
- Added log sanitization
- Removed all hardcoded credentials
