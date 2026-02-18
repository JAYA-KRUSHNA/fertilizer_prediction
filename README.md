# 🌾 AgriFert Predict

**Smart Fertilizer Management System for Sustainable Agriculture**

AgriFert Predict is an AI-powered web application that revolutionizes fertilizer management in agriculture. Using advanced machine learning algorithms, it analyzes soil properties, crop types, and environmental factors to provide precise fertilizer recommendations, helping farmers optimize resource use and improve sustainability.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

### 🤖 AI-Powered Predictions
- **Machine Learning Models**: Random Forest, Support Vector Regression, and Linear Regression
- **Feature Importance Analysis**: Understand which factors most influence predictions
- **Real-time Processing**: Instant fertilizer recommendations

### 🌐 Modern Web Interface
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Interactive 3D Visualizations**: Three.js-powered agricultural scene
- **Accessibility Compliant**: WCAG 2.1 AA compliant design
- **Progressive Web App**: Installable PWA for offline use

### 📊 Comprehensive Analytics
- **Data Visualization Dashboard**: Interactive charts and graphs
- **Model Performance Comparison**: Compare different ML algorithms
- **Prediction History**: Track and analyze past recommendations
- **Statistical Insights**: Dataset overview and key metrics

### 🗄️ Data Management
- **SQLite Database**: Efficient local data storage
- **Synthetic Dataset**: 10,000+ realistic agricultural samples
- **Data Validation**: Comprehensive input validation and error handling
- **Export Capabilities**: Download results and reports

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, Flask 2.3.3
- **Machine Learning**: scikit-learn 1.3.0, pandas, numpy
- **Database**: SQLite3
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualization**: Chart.js, Three.js, Plotly
- **Deployment**: Ready for Docker/Gunicorn

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning repository)

## 🚀 Installation

1. **Clone the repository** (optional):
   ```bash
   git clone https://github.com/yourusername/agrifert-predict.git
   cd agrifert-predict
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run data generation**:
   ```bash
   python data_generator.py
   ```

5. **Clean and preprocess data**:
   ```bash
   python data_cleaner.py
   ```

6. **Train the machine learning models**:
   ```bash
   python model_trainer.py
   ```

7. **Start the Flask application**:
   ```bash
   python app.py
   ```

8. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## 📖 Usage

### Making Predictions

1. **Navigate to the Prediction Page**: Click "Predict" in the navigation menu
2. **Fill in the Form**:
   - Enter soil properties (pH, texture, nutrient levels)
   - Select crop type and irrigation method
   - Input climate conditions (temperature, humidity, rainfall)
3. **Get Recommendations**: Click "Calculate" to receive AI-powered fertilizer suggestions

### Understanding Results

- **Fertilizer Amount**: Recommended fertilizer quantity in kg/ha
- **Key Factors**: Which parameters most influenced the prediction
- **Environmental Impact**: Sustainability assessment
- **Application Tips**: Best practices for fertilizer application

### Analytics Dashboard

- **Overview Tab**: Dataset statistics and key metrics
- **Model Performance**: Compare different ML algorithms
- **Feature Analysis**: Understand variable importance
- **Prediction History**: Review past recommendations

## 📁 Project Structure

```
agri_fertilizer_prediction/
├── app.py                          # Main Flask application
├── data_generator.py              # Synthetic data generation
├── data_cleaner.py                # Data preprocessing
├── model_trainer.py               # ML model training
├── database_setup.py              # Database operations
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── templates/                     # HTML templates
│   ├── home.html
│   ├── predict.html
│   ├── results.html
│   ├── visualization.html
│   └── about.html
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css             # Main stylesheet
│   ├── js/
│   │   ├── main.js              # Frontend interactivity
│   │   └── three_scene.js       # 3D visualizations
│   └── images/                   # Image assets
├── models/                        # Trained ML models
├── data/                          # Dataset files
└── __pycache__/                   # Python cache
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///data/agricultural_data.db
MODEL_PATH=models/best_model.pkl
SCALER_PATH=models/scaler.pkl
```

### Model Configuration

Modify `model_trainer.py` to adjust:
- Training parameters
- Model hyperparameters
- Cross-validation settings
- Feature selection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Agricultural Research**: Based on comprehensive soil science and agronomy research
- **Open Source Libraries**: Built with Flask, scikit-learn, pandas, and other amazing libraries
- **Three.js Community**: For the incredible 3D visualization capabilities
- **Agricultural Experts**: For providing domain knowledge and validation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/agrifert-predict/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agrifert-predict/discussions)
- **Email**: jayakrushna1622@gmail.com

## 🔮 Future Enhancements

- [ ] **Mobile App**: React Native mobile application
- [ ] **IoT Integration**: Sensor data integration for real-time monitoring
- [ ] **Weather API**: Real-time weather data integration
- [ ] **Multi-language Support**: Internationalization (i18n)
- [ ] **Advanced Analytics**: Time-series analysis and forecasting
- [ ] **API Endpoints**: RESTful API for third-party integrations
- [ ] **Cloud Deployment**: AWS/GCP/Azure deployment configurations

---

**Made with ❤️ for sustainable agriculture**

*Optimize fertilizer use, maximize yields, protect the environment*
