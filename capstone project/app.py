from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io
import base64

app = Flask(__name__)

# Load sample data
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Marketing_Spend': [2000, 2500, 2200, 2700, 3000, 3500, 3300, 3100, 3000, 3200, 4000, 4500],
    'Sales': [15000, 17000, 16500, 18000, 20000, 22000, 21500, 21000, 20500, 22500, 25000, 28000]
}
df = pd.DataFrame(data)

# Build model
X = df[['Marketing_Spend']]
y = df['Sales']
model = LinearRegression().fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    spend = float(request.form['spend'])
    prediction = model.predict([[spend]])[0]
    fig, ax = plt.subplots()
    sns.scatterplot(x='Marketing_Spend', y='Sales', data=df, ax=ax)
    ax.plot([spend], [prediction], marker='o', color='red', label='Prediction')
    ax.legend()
    plt.xlabel('Marketing Spend ($)')
    plt.ylabel('Sales ($)')
    plt.title('Marketing Spend vs Sales')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('result.html', prediction=prediction, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True,port=12)
