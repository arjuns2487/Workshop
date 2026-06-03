# House Price Prediction

A web-based machine learning application that predicts house prices based on property characteristics using linear regression.

## Overview

This project combines a Python Flask backend with a clean, responsive web interface to provide real-time house price predictions. The prediction model is trained on historical housing data and uses property features like area, number of bedrooms, and age to estimate market prices.

## Features

- 🏠 **Intuitive Web Interface** - User-friendly form to input property details
- 🤖 **Machine Learning Model** - Linear regression model trained on historical data
- ⚡ **Real-time Predictions** - Get instant price estimates with a single click
- 📊 **Data Preprocessing** - Automatic handling of missing values
- 🎨 **Responsive Design** - Works seamlessly on desktop and mobile devices

## Tech Stack

- **Backend**: Python 3 with Flask
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Machine Learning**: Scikit-learn (Linear Regression)
- **Data Processing**: Pandas

## Project Structure

```
├── app.py                      # Flask application and ML model
├── homeprices.csv             # Training dataset with house prices
├── templates/
│   └── index.html            # Main web interface
├── static/
│   └── style.css             # Styling and layout
└── README.md                  # This file
```

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/arjuns2487/Workshop.git
   cd Workshop
   ```

2. **Install dependencies**
   ```bash
   pip install flask pandas scikit-learn
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   Navigate to http://localhost:5000
   ```

## Usage

1. Open the web application in your browser
2. Enter the property details:
   - **Area (sqft)**: Total square footage of the property
   - **Bedrooms**: Number of bedrooms
   - **Age (years)**: Age of the home in years
3. Click the "Submit" button
4. View the predicted price displayed below the form

## Dataset

The model is trained on `homeprices.csv` containing:
- **area**: Property size in square feet
- **bedrooms**: Number of bedrooms
- **age**: Age of the property in years
- **price**: Actual market price (target variable)

The model automatically handles missing values by using the median bedroom count from the dataset.

## Model Details

- **Algorithm**: Linear Regression
- **Training Samples**: 6 properties
- **Features**: 3 (area, bedrooms, age)
- **Missing Value Strategy**: Median imputation for bedrooms

## Future Enhancements

- [ ] Expand dataset for improved accuracy
- [ ] Add more features (location, condition, parking, etc.)
- [ ] Implement advanced models (Random Forest, Gradient Boosting)
- [ ] Add data visualization (price trends, feature importance)
- [ ] Database integration for historical predictions
- [ ] Deploy to cloud platform (Heroku, AWS, etc.)
- [ ] Add input validation and error handling
- [ ] Create API documentation

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is open source and available under the MIT License.

## Author

[arjuns2487](https://github.com/arjuns2487)

## Disclaimer

This model is for educational purposes only. Actual real estate valuations should involve professional appraisals and multiple factors not captured in this simplified model.

---

**Created**: December 2025  
**Last Updated**: December 19, 2025
