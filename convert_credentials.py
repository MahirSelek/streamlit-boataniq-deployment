"""
Helper script to convert JSON credentials to Streamlit secrets format
Run this locally to generate the secrets.toml format
"""

import json
import os

def convert_json_to_toml(json_path, output_path=None):
    """
    Convert GCP JSON credentials to Streamlit secrets TOML format
    
    Args:
        json_path: Path to your JSON credentials file
        output_path: Optional path to save the TOML file (default: secrets_example.toml)
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        creds = json.load(f)
    
    # Generate TOML format
    toml_content = "[gcp_credentials]\n"
    toml_content += f'type = "{creds["type"]}"\n'
    toml_content += f'project_id = "{creds["project_id"]}"\n'
    toml_content += f'private_key_id = "{creds["private_key_id"]}"\n'
    
    # Handle private key (needs special formatting)
    private_key = creds["private_key"].replace('\n', '\\n')
    toml_content += f'private_key = "{private_key}"\n'
    
    toml_content += f'client_email = "{creds["client_email"]}"\n'
    toml_content += f'client_id = "{creds["client_id"]}"\n'
    toml_content += f'auth_uri = "{creds["auth_uri"]}"\n'
    toml_content += f'token_uri = "{creds["token_uri"]}"\n'
    toml_content += f'auth_provider_x509_cert_url = "{creds["auth_provider_x509_cert_url"]}"\n'
    toml_content += f'client_x509_cert_url = "{creds["client_x509_cert_url"]}"\n'
    
    # Save to file
    if output_path is None:
        output_path = "secrets_example.toml"
    
    with open(output_path, 'w') as f:
        f.write(toml_content)
    
    print(f"✅ Converted credentials to {output_path}")
    print("\n⚠️  IMPORTANT: This file contains sensitive information!")
    print("   - DO NOT commit this file to GitHub")
    print("   - Use this format in Streamlit Cloud secrets")
    print("   - Delete this file after copying to Streamlit Cloud")
    
    return toml_content

if __name__ == "__main__":
    # Path to your credentials file
    json_file = "../static-chiller-472906-f3-4ee4a099f2f1.json"
    
    if os.path.exists(json_file):
        convert_json_to_toml(json_file)
        print("\n📋 Copy the content above to Streamlit Cloud secrets")
    else:
        print(f"❌ Credentials file not found: {json_file}")
        print("   Please update the path in this script")
