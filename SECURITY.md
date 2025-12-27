# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project:

1. **Do NOT open a public issue**
2. Email the maintainer directly or use GitHub's private vulnerability reporting feature
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Resolution**: Depends on severity

## Scope

This is a local simulation tool with no network functionality. Security concerns are primarily:

- Malicious YAML configuration files (handled by `yaml.safe_load`)
- Path traversal in file operations

Thank you for helping keep this project safe.
