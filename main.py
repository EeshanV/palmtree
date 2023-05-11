from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import confusion_matrix, accuracy_score

app = Flask(__name__)

df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global df
    file = request.files['file']
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    # You can add support for other file formats (e.g., databases) here

    description = df.describe().to_html()
    return render_template('index.html', description=description, df_available=True)

@app.route('/head', methods=['POST'])
def head():
    global df
    if df is not None:
        head_df = df.head().to_html()
        return render_template('index.html', head_df=head_df, df_available=True)
    else:
        return render_template('index.html', df_available=False)


@app.route('/boxplot', methods=['POST'])
def boxplot():
    global df
    if df is not None:
        plt.boxplot(df)
        plt.title('Box Plot')
        plt.xlabel('Columns')
        plt.ylabel('Values')

        plot_io = BytesIO()
        plt.savefig(plot_io, format='png')
        plot_io.seek(0)
        plot_data = base64.b64encode(plot_io.getvalue()).decode('utf-8')

        plt.clf()

        return render_template('index.html', boxplot=plot_data, df_available=True)
    else:
        return render_template('index.html', df_available=False)
    
@app.route('/classification', methods=['POST'])
def clasification():
    global df
    x= df.iloc[::,0:-1]
    y = df.iloc[::,-1]
    if df is not None:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        classifier = lr()
        model = classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        cm = confusion_matrix(y_test,y_pred) 
        op = f'Classification matrix : \n\n {cm} \n\nAccuracy score : {accuracy_score(y_test,y_pred)}'
        return render_template('index.html', classification_result=op, df_available=True)
    else:
        return render_template('index.html', df_available=False)
        


if __name__ == '__main__':
    app.run(debug=True)