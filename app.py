from flask import Flask,request
from flask import render_template
from lstm import stockpredict
from eurocall import computecalloptionBS,computecalloptionTri,computecalloptionAMM
from americancall import computeAmericanBS, computeAmericanMC

app = Flask(__name__)

@app.route('/')
def index():
    my_list = [{"value": "TSLA"},
               {"value": "BIDU"},
               {"value": "AMZN"},
               {"value": "BABA"},
               {"value": "GOOG"}]
    return render_template('index.html', title='hello world', my_list=my_list)


@app.route("/predict", methods=['GET','POST'])
def predict():
    stock = request.form.get('form-stock')
    return stockpredict(stock)

@app.route("/european", methods=['GET','POST'])
def eurocall():
    S0 = float(request.form.get('s'))
    K = float(request.form.get('k'))
    rf = float(request.form.get('rf'))
    divR = float(request.form.get('divR'))
    sigma = float(request.form.get('sigma'))
    T = float(request.form.get('T'))
    H = float(request.form.get('H'))
    if request.form["euromethod"]=='tt':
        N = int(request.form.get('N'))
        return render_template('index.html', title='hello world', my_list='',
                               downandin=str(computecalloptionTri(S0, K, rf, divR, sigma, T, N, H)[0]),
                               downandout=str(computecalloptionTri(S0, K, rf, divR, sigma, T, N, H)[1]))
    if request.form["euromethod"]=='amm':
        M = int(request.form.get('M'))
        return render_template('index.html', title='hello world', my_list='',
                               downandin='',
                               downandout=str(computecalloptionAMM(S0, K, rf, divR, sigma, T, M, H)))
    if request.form["euromethod"] == 'bs':
        return render_template('index.html', title='hello world', my_list='',
                               downandin=str(computecalloptionBS(S0, K, rf, divR, sigma, T, H)[0]),
                               downandout=str(computecalloptionBS(S0, K, rf, divR, sigma, T, H)[1]))

@app.route("/american", methods=['GET','POST'])
def americancall():
    S0 = float(request.form.get('s_a'))
    K = float(request.form.get('k_a'))
    T = float(request.form.get('t_a'))
    M = int(request.form.get('m_a'))
    r = float(request.form.get('r_a'))
    delta = float(request.form.get('delta'))
    sigma = float(request.form.get('sigma_a'))
    i = int(request.form.get('i'))
    seed = int(request.form.get('seed'))
    if request.form["americanmethod"]=='mc':
        return render_template('index.html', title='hello world', my_list='',
                               calloption=str(computeAmericanMC(S0, K, T, M, r, delta, sigma, i,seed, 'call')),
                               putoption=str(computeAmericanMC(S0, K, T, M, r, delta, sigma, i,seed, 'put')))
    if request.form["americanmethod"]=='bs':
        return render_template('index.html', title='hello world', my_list='',
                               calloption=str(computeAmericanBS(S0, K, T, M, r, delta, sigma, i, 'call')),
                               putoption=str(computeAmericanBS(S0, K, T, M, r, delta, sigma, i, 'put')))

if __name__ == '__main__':
    app.run()