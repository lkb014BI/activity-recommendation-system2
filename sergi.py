from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from numpy.linalg import svd

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = None
    if request.method == 'POST':
        liked_activity = request.form['liked_activity']
        dades = pd.read_csv('data.txt', delimiter=',')
        UtlMtrx = dades.pivot_table(
            values='score', index='user_id', columns='act_id', fill_value=0)
        X = UtlMtrx.T
        U, S, Vt = svd(X, full_matrices=False)
        resultant_matrix = U @ np.diag(S)
        corrMtx = np.corrcoef(resultant_matrix)
        names = UtlMtrx.columns
        names_list = list(names)
        try:
            id_liked = names_list.index(liked_activity)
        except ValueError:
            return render_template('index.html', recommendation="Activity not found!")
        corr_recom = corrMtx[id_liked]
        ids = (corr_recom > 0.9) & (corr_recom < 0.999)
        tmp = []
        for i in range(len(names[ids])):
            myTuple = (corr_recom[ids][i].round(3), names[ids][i])
            x = " - ".join(map(str, myTuple))
            tmp.append(x)
        tmp.sort(reverse=True)
        recommendation = tmp
    return render_template('index.html', recommendation=recommendation)


if __name__ == '__main__':
    app.run(debug=True)
