# BoataniQ Streamlit Deployment

This folder contains everything needed to deploy the BoataniQ boat analysis app to Streamlit Cloud.

## 🚀 Quick Start Guide

### Step 1: Create a GitHub Repository

1. **Initialize Git in this folder:**
   ```bash
   cd streamlit-deployment
   git init
   git add .
   git commit -m "Initial commit: Streamlit deployment"
   ```

2. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Create a **public** repository (e.g., `boataniq-streamlit`)
   - **DO NOT** initialize with README, .gitignore, or license

3. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/boataniq-streamlit.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Set Up Streamlit Secrets

**IMPORTANT:** Your credentials file (`static-chiller-472906-f3-4ee4a099f2f1.json`) should NEVER be committed to GitHub!

Instead, you'll use Streamlit's secrets management:

1. **Convert your JSON credentials to a format for Streamlit:**
   - Open your JSON file: `static-chiller-472906-f3-4ee4a099f2f1.json`
   - Copy the entire JSON content

2. **In Streamlit Cloud:**
   - Go to your app settings
   - Click on "Secrets" in the left sidebar
   - Add the following structure:
   ```toml
   [gcp_credentials]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "your-private-key-id"
   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
   client_email = "your-service-account@your-project.iam.gserviceaccount.com"
   client_id = "your-client-id"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
   ```
   
   **OR** you can paste the entire JSON as a single string (the app will parse it):
   ```toml
   [gcp_credentials]
   # Paste your entire JSON credentials here as a single string
   # The app will parse it automatically
   ```

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Deploy your app:**
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/boataniq-streamlit`
   - Set Main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Configure secrets:**
   - After deployment, go to app settings
   - Add your GCP credentials in the Secrets section (as described in Step 2)

### Step 4: Verify Deployment

1. Your app should be live at: `https://YOUR_APP_NAME.streamlit.app`
2. Upload a boat image to test
3. Verify that analysis works correctly

## 📁 Project Structure

```
streamlit-deployment/
├── streamlit_app.py              # Main Streamlit application
├── boat_vertex_ai_analyzer.py    # AI analyzer (modified for env vars)
├── image_preprocessor.py         # Image preprocessing and validation
├── boat_database.py              # Boat database (optional, for future features)
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── .gitignore                    # Git ignore file (excludes credentials)
└── README.md                     # This file
```

## 🔒 Security Notes

1. **Never commit credentials:**
   - The `.gitignore` file is configured to exclude all `.json` files
   - Your credentials file should remain on your local machine only

2. **Use Streamlit Secrets:**
   - All sensitive data should be stored in Streamlit Cloud secrets
   - Never hardcode credentials in your code

3. **Public Repository:**
   - Since this is a public repo, make sure no secrets are in the code
   - All credentials are handled through Streamlit's secrets management

## 🛠️ Local Development

To run locally:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up local secrets (optional):**
   - Create `.streamlit/secrets.toml` (this file is gitignored)
   - Add your credentials there for local testing

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📝 Environment Variables

The app uses Streamlit secrets to access GCP credentials. The structure should be:

```toml
[gcp_credentials]
# Your entire JSON credentials as a dictionary
type = "service_account"
project_id = "..."
# ... etc
```

## 🐛 Troubleshooting

### "GCP credentials not found"
- Make sure you've added credentials to Streamlit Cloud secrets
- Check that the secret key is `gcp_credentials`
- Verify the JSON structure is correct

### "Failed to initialize Vertex AI"
- Check that your service account has the necessary permissions
- Verify the project ID is correct
- Ensure Vertex AI API is enabled in your GCP project

### Image analysis fails
- Make sure the image is clear and shows a boat
- Check that the image format is supported (PNG, JPG, JPEG, GIF, WEBP)
- Verify your GCP quota hasn't been exceeded

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## 🎯 Next Steps

After deployment, you can:
- Add boat database integration (if you have CSV data)
- Implement user authentication
- Add analysis history storage
- Integrate with external APIs
- Add more advanced features

## 📄 License

This project is for your use. Make sure to comply with Google Cloud's terms of service and Streamlit's usage policies.
