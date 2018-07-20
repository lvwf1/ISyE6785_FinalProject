from flask import Flask,request
from flask import render_template
from lstm import stockpredict

app = Flask(__name__)

@app.route('/')
def index():
    list1 = list(range(10))
    my_list = [{"value": "Tesla"},
               {"value": "BIDU"},
               {"value": "Amazon"},
               {"value": "Alibaba"},
               {"value": "Google"}]
    return render_template('index.html', title='hello world', my_list=my_list)


@app.route("/predict", methods=['GET','POST'])
def predict():
    stock = request.form.get('form-stock')
    return stockpredict(stock)

if __name__ == '__main__':
    app.run()

def predict(stock):
    return stock