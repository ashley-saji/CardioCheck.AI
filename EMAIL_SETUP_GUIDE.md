# 📧 Email Setup Guide for Heart Disease Prediction App

## ✅ **Current Status**: Email functionality improved with better error handling and configuration options

## 🛠️ **How to Set Up Email**

### **Option 1: Configure in the Web Interface (Recommended)**

1. **Open the Streamlit app**: `http://localhost:8501`
2. **In the sidebar**, scroll down to "📧 Email Configuration"
3. **Expand the section** and enter:
   - **Sender Email**: Your Gmail address (e.g., `yourname@gmail.com`)
   - **App Password**: Your Gmail App Password (NOT your regular password)
4. **Test the configuration** with the "🧪 Test Email" button

### **Option 2: Configure in Code (For permanent setup)**

Edit `streamlit_app.py` around line 746:

```python
# Email configuration - Use configured values or defaults
sender_email = "YOUR_GMAIL_ADDRESS@gmail.com"  # Replace this
sender_password = "YOUR_16_DIGIT_APP_PASSWORD"  # Replace this
```

## 🔑 **Gmail App Password Setup**

### **Step 1: Enable 2-Factor Authentication**
1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Under "Signing in to Google", click "2-Step Verification"
3. Follow the setup process

### **Step 2: Generate App Password**
1. In Google Account Security, click "2-Step Verification"
2. At the bottom, click "App passwords"
3. Select "Mail" and your device
4. Click "Generate"
5. **Copy the 16-digit password** (e.g., `abcd efgh ijkl mnop`)

### **Step 3: Use App Password**
- **NOT your regular Gmail password**
- **Use the 16-digit App Password**
- **Enter without spaces**

## 🧪 **Testing Email Configuration**

### **Method 1: Use Built-in Test**
1. Configure email credentials in sidebar
2. Enter a test email address
3. Click "🧪 Test Email" button
4. Check if test email is received

### **Method 2: Send Full Report**
1. Complete a heart disease prediction
2. Enter patient email address
3. Click "📧 Email Report"
4. Check for success/error messages

## ❌ **Common Issues & Solutions**

### **Issue**: "Authentication failed"
**Solution**: 
- Use App Password, not regular password
- Ensure 2FA is enabled
- Check email address is correct

### **Issue**: "SMTP connection failed"
**Solution**:
- Check internet connection
- Verify firewall isn't blocking port 587
- Try different network if corporate firewall blocks SMTP

### **Issue**: "Invalid credentials"
**Solution**:
- Regenerate App Password
- Copy password exactly (no spaces)
- Verify email address is correct

### **Issue**: "Email not configured"
**Solution**:
- Enter valid email credentials in sidebar
- Or edit the code with permanent values
- Check that fields aren't still showing defaults

## 🔧 **Alternative Email Services**

If Gmail doesn't work, you can modify the SMTP settings:

### **Outlook/Hotmail**:
```python
smtp_server = "smtp-mail.outlook.com"
smtp_port = 587
```

### **Yahoo Mail**:
```python
smtp_server = "smtp.mail.yahoo.com"
smtp_port = 587
```

### **Custom SMTP**:
```python
smtp_server = "your.smtp.server.com"
smtp_port = 587  # or 465 for SSL
```

## 📋 **Email Features**

### **What Gets Sent**:
- Professional PDF report attachment
- Patient information and results
- Medical recommendations
- Proper disclaimers

### **Email Template**:
- Clean, professional format
- Patient name personalization
- Date and system information
- Medical disclaimer

## 🚨 **Security Notes**

### **App Passwords**:
- More secure than regular passwords
- Can be revoked independently
- Specific to each application

### **Data Privacy**:
- Emails contain sensitive medical information
- Ensure secure email transmission
- Follow HIPAA guidelines if applicable

## ✅ **Quick Verification Checklist**

- [ ] 2FA enabled on Gmail
- [ ] App Password generated
- [ ] Correct email and password entered
- [ ] Test email sent successfully
- [ ] PDF report generation works
- [ ] Email delivery confirmed

## 📞 **Still Having Issues?**

### **Error Messages to Watch For**:
1. "Email not configured" → Enter credentials in sidebar
2. "Authentication failed" → Check App Password
3. "Connection refused" → Check network/firewall
4. "Invalid email" → Verify email format

### **Debug Steps**:
1. Try the test email function first
2. Check Streamlit console for detailed errors
3. Verify Gmail settings in web interface
4. Test with a different email address

## 🎯 **Summary**

The email functionality now includes:
- ✅ Configurable credentials in web interface
- ✅ Test email functionality
- ✅ Better error messages and guidance
- ✅ Detailed setup instructions
- ✅ Fallback options for different email providers

**The email system is working**, but requires proper Gmail App Password configuration to send emails successfully!