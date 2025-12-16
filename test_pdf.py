"""
Test script to verify PDF generation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_app import HeartDiseaseWebApp
from datetime import datetime

def test_pdf_generation():
    """Test PDF generation functionality"""
    print("🧪 Testing PDF generation...")
    
    # Create app instance
    app = HeartDiseaseWebApp()
    
    # Sample prediction results
    sample_results = {
        'prediction': 1,
        'probability': 0.75,
        'risk_level': 'High Risk',
        'confidence': 'High Confidence',
        'percentage': '75.0%'
    }
    
    # Sample features
    sample_features = {
        'age': 60,
        'sex': 1,
        'cp': 2,
        'trestbps': 150,
        'chol': 280,
        'fbs': 1,
        'restecg': 1,
        'thalch': 120,
        'exang': 1,
        'oldpeak': 2.0,
        'slope': 1,
        'ca': 2,
        'thal': 2
    }
    
    # Sample patient info
    sample_patient_info = {
        'name': 'John Doe',
        'email': 'john.doe@example.com',
        'doctor': 'Dr. Smith',
        'age': 60,
        'sex': 'Male'
    }
    
    try:
        # Test PDF generation
        pdf_buffer = app.generate_pdf_report(sample_results, sample_features, sample_patient_info)
        
        if pdf_buffer:
            print("✅ PDF generation successful!")
            
            # Save to file for testing
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"test_report_{timestamp}.pdf"
            
            with open(filename, 'wb') as f:
                f.write(pdf_buffer.getvalue())
            
            print(f"✅ Test PDF saved as: {filename}")
            return True
        else:
            print("❌ PDF generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during PDF generation: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_pdf_generation()
    if success:
        print("\n🎉 PDF generation test passed!")
    else:
        print("\n❌ PDF generation test failed!")