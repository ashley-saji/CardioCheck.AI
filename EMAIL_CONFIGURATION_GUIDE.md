# 📧 Email Configuration System - User Guide

## Overview
The Heart Disease Prediction app now includes a **persistent email configuration system** that saves your Gmail settings so you don't need to re-enter them every time you open the app.

## 🔧 New Features

### **1. Automatic Configuration Saving**
- Email settings are saved locally in `email_config.json`
- Password is encoded for basic security
- Configuration persists between app sessions

### **2. Enhanced UI in Sidebar**
- **Status Display**: Shows if email is configured
- **Save Button**: Stores configuration permanently
- **Test Button**: Verifies email functionality
- **Clear Button**: Removes saved configuration

### **3. Security Features**
- Passwords are base64 encoded (not plain text)
- Configuration file is local only
- Easy to clear when not needed

## 📋 How to Use

### **First Time Setup:**
1. Open the app and go to sidebar
2. Expand "📧 Email Configuration" section
3. Enter your Gmail address
4. Enter your Gmail App Password (not regular password)
5. Enter your name (optional)
6. Click "💾 Save Config"
7. Click "🧪 Test Email" to verify

### **Subsequent Uses:**
1. Open the app - configuration loads automatically
2. Email settings are pre-filled
3. No need to re-enter credentials
4. Just use the email functionality

### **Managing Configuration:**
- **View Status**: Check if email is configured in sidebar
- **Update Settings**: Enter new details and click "Save Config"
- **Test Connection**: Use "Test Email" button anytime
- **Remove Settings**: Click "Clear Config" to delete

## 🔒 Security Notes

### **What's Stored:**
```json
{
  "email": "your-email@gmail.com",
  "password": "base64_encoded_password",
  "name": "Your Name",
  "saved_date": "2025-09-22T..."
}
```

### **Security Measures:**
- Password is base64 encoded (not plain text)
- File is stored locally only
- No data sent to external servers
- Easy to delete configuration

### **Recommendations:**
- Clear config when not using the app
- Use Gmail App Passwords only
- Don't share the configuration file

## 📱 Gmail App Password Setup

### **Step-by-Step:**
1. **Enable 2FA on Gmail**
   - Go to Google Account Settings
   - Security → 2-Step Verification
   - Follow setup instructions

2. **Generate App Password**
   - In Security settings
   - App Passwords → Select app
   - Choose "Mail" or "Other"
   - Copy the generated password

3. **Use in App**
   - Enter Gmail address
   - Enter App Password (16 characters)
   - Save configuration

## 🚀 Benefits

### **For Users:**
- ✅ No repeated configuration
- ✅ Faster email sending
- ✅ Better user experience
- ✅ Secure local storage

### **For Developers:**
- ✅ Better error handling
- ✅ Improved user feedback
- ✅ Persistent configuration
- ✅ Enhanced security

## 🛠️ Troubleshooting

### **Common Issues:**

**"Email configuration not found"**
- Solution: Configure email in sidebar first

**"Authentication failed"**
- Check App Password is correct
- Ensure 2FA is enabled
- Verify Gmail address

**"Test email failed"**
- Check internet connection
- Verify Gmail settings
- Try generating new App Password

**"Configuration not saving"**
- Check file permissions
- Ensure app has write access
- Try running as administrator

### **Reset Configuration:**
1. Click "🗑️ Clear Config" in sidebar
2. Or manually delete `email_config.json`
3. Restart app and reconfigure

## 📂 File Structure

```
web app final/
├── streamlit_app.py (updated)
├── email_config.json (created automatically)
└── ... (other files)
```

## 🔄 Configuration File Format

```json
{
  "email": "doctor@gmail.com",
  "password": "YWJjZGVmZ2hpams=",
  "name": "Dr. Smith",
  "saved_date": "2025-09-22T10:30:00"
}
```

## ✅ Quick Start Checklist

- [ ] Open Heart Disease Prediction app
- [ ] Go to sidebar → Email Configuration
- [ ] Enter Gmail address
- [ ] Enter Gmail App Password
- [ ] Click "Save Config"
- [ ] Click "Test Email"
- [ ] Verify test email received
- [ ] Configuration now persistent!

## 📞 Support

If you encounter issues:
1. Check Gmail App Password setup
2. Verify 2FA is enabled
3. Try clearing and reconfiguring
4. Check internet connection
5. Restart the application

---

**Note**: This system uses local file storage for simplicity. For production deployments, consider using secure cloud-based configuration management.