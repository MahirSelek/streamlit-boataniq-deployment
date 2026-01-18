# 🚀 Step-by-Step Deployment Guide

This guide will walk you through deploying BoataniQ to Streamlit Cloud step by step.

## Prerequisites

- ✅ Your credentials file: `static-chiller-472906-f3-4ee4a099f2f1.json`
- ✅ A GitHub account
- ✅ A Streamlit Cloud account (free)

---

## Step 1: Prepare Your Repository

### 1.1 Navigate to the deployment folder
```bash
cd /Users/mahirselek/Desktop/DSPhD/MS/denizmen-scraping/streamlit-deployment
```

### 1.2 Initialize Git (if not already done)
```bash
git init
git add .
git commit -m "Initial commit: Streamlit deployment ready"
```

### 1.3 Verify .gitignore is working
Make sure your credentials file is NOT in this folder. The `.gitignore` should prevent any `.json` files from being committed.

---

## Step 2: Create GitHub Repository

### 2.1 Create a new repository on GitHub
1. Go to https://github.com/new
2. Repository name: `boataniq-streamlit` (or any name you prefer)
3. Description: "AI-powered boat analysis app"
4. **Make it PUBLIC** (required for free Streamlit Cloud)
5. **DO NOT** check "Add a README file"
6. **DO NOT** check "Add .gitignore"
7. Click "Create repository"

### 2.2 Connect and push your code
```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/boataniq-streamlit.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

**✅ Check:** Go to your GitHub repository and verify all files are there (except `.json` files).

---

## Step 3: Set Up Streamlit Cloud

### 3.1 Sign in to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click "Sign in" and authorize with your GitHub account

### 3.2 Deploy your app
1. Click "New app" button
2. Fill in the form:
   - **Repository:** Select `YOUR_USERNAME/boataniq-streamlit`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** (auto-generated, or choose custom)
3. Click "Deploy"

### 3.3 Wait for initial deployment
- Streamlit will install dependencies and deploy
- This may take 2-3 minutes
- You'll see build logs in real-time

---

## Step 4: Configure Secrets (CRITICAL!)

### 4.1 Open your credentials file
Open: `/Users/mahirselek/Desktop/DSPhD/MS/denizmen-scraping/static-chiller-472906-f3-4ee4a099f2f1.json`

### 4.2 In Streamlit Cloud
1. Go to your app's settings (click the "⋮" menu → "Settings")
2. Click "Secrets" in the left sidebar
3. You'll see a text editor

### 4.3 Add your credentials
Copy the ENTIRE content of your JSON file and paste it into the secrets editor in this format:

```toml
[gcp_credentials]
type = "service_account"
project_id = "static-chiller-472906"
private_key_id = "your-private-key-id-here"
private_key = """-----BEGIN PRIVATE KEY-----
YOUR_PRIVATE_KEY_HERE
-----END PRIVATE KEY-----"""
client_email = "your-service-account@static-chiller-472906.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```

**OR** (easier method) - just paste your entire JSON as a string:

```toml
[gcp_credentials]
# Paste your entire JSON file content here as a single string
# The app will parse it automatically
```

**Actually, the easiest way:**
1. Open your JSON file
2. Copy everything inside it (the JSON object)
3. In Streamlit secrets, paste it like this:

```toml
[gcp_credentials]
type = "service_account"
project_id = "static-chiller-472906"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

**Important:** 
- For `private_key`, keep the newlines as `\n` or use triple quotes `"""..."""`
- Make sure all values are properly quoted

### 4.4 Save secrets
Click "Save" at the bottom of the secrets editor.

---

## Step 5: Restart Your App

1. Go back to your app
2. Click the "⋮" menu → "Rerun app"
3. Wait for the app to restart with new secrets

---

## Step 6: Test Your App

1. Your app should be live at: `https://YOUR_APP_NAME.streamlit.app`
2. Upload a boat image
3. Click "Analyze Boat"
4. Verify that analysis works

---

## Troubleshooting

### ❌ "GCP credentials not found"
- **Solution:** Make sure you saved the secrets in Streamlit Cloud
- Check that the secret key is exactly `[gcp_credentials]`
- Verify the JSON structure matches your credentials file

### ❌ "Failed to initialize Vertex AI"
- **Solution:** 
  - Check that Vertex AI API is enabled in your GCP project
  - Verify your service account has the necessary permissions
  - Check that the project ID in secrets matches your GCP project

### ❌ "Image analysis fails"
- **Solution:**
  - Make sure the image is clear and shows a boat
  - Check image format (PNG, JPG, JPEG, GIF, WEBP)
  - Verify your GCP quota hasn't been exceeded

### ❌ Build fails
- **Solution:**
  - Check `requirements.txt` is correct
  - Verify all dependencies are listed
  - Check build logs for specific errors

---

## Security Checklist

Before making your repo public, verify:

- ✅ No `.json` files are in the repository
- ✅ No hardcoded credentials in code
- ✅ `.gitignore` includes `*.json`
- ✅ All secrets are in Streamlit Cloud only
- ✅ Credentials file remains on your local machine only

---

## Updating Your App

To update your app:

1. Make changes to your code locally
2. Commit and push:
   ```bash
   git add .
   git commit -m "Your update message"
   git push
   ```
3. Streamlit Cloud will automatically redeploy

---

## Next Steps

After successful deployment:

1. ✅ Share your app URL with others
2. ✅ Monitor usage in Streamlit Cloud dashboard
3. ✅ Add more features as needed
4. ✅ Consider adding boat database integration
5. ✅ Implement user authentication (optional)

---

## Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify secrets are correctly formatted
3. Test locally first with `streamlit run streamlit_app.py`
4. Check Google Cloud Console for API quotas/errors

---

**🎉 Congratulations!** Your app should now be live and accessible to everyone!
