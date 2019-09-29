from textwrap import dedent
import json

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import joblib
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objs as go

from components import Column, Header, Row

# Loading stored data/models
lda = joblib.load('data/lda.model')
tf_vectorizer = joblib.load('data/tf_vectorizer.model')
df = pd.read_csv('data/arxiv_data.csv')

for i in range(10):
    df.id = df.id.apply(lambda x: x.replace(f'v{i}', ''))

# Prelim functions
idx2words = {v: k for k, v in tf_vectorizer.vocabulary_.items()}

# Some global data vars
tf = tf_vectorizer.fit_transform(df.summary)
tm = lda.transform(tf)


def card_style(background='#626efa', font='#e5ecf6'):
    return {
        'background-color': background,
        'height': '40%',
        'padding': '2px 5px 2px 5px',
        'color': font,
        'margin': '2px 5px 2px 5px',
    }


def get_top_words(topic, k=10):
    top_idx = topic.argsort()[:-k-1:-1]
    top_words = [idx2words[idx] for idx in top_idx]

    return top_words


def get_top_topics(mixture, k=5):
    top_topics = mixture.argsort()[:-k-1:-1]
    return list(top_topics), list(mixture[top_topics])


# Start layout
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for deployments

# Standard Dash app code below
app.layout = html.Div(className='container', children=[

    Header('Arxiv Assistant', app),

    Row([
        dcc.Input(
            id='input-url',
            placeholder='Enter URL...',
            type='text',
            value='',
            style={
                'width': '80%',
                'margin-right': '20px'
            }
        ),
        html.Button('Submit', id='button-update')
    ]),
    Row([
        Column(width=6, children=[
            dcc.Markdown(id='markdown-article', style=card_style()),
            dcc.Markdown(id='markdown-recommendations', style=card_style('#e5ecf6', '#626efa'))
        ]),
        Column(width=6, children=[
            dcc.Graph(id='graph-topics', style={'height': '40%'}),
            dcc.Markdown(id='markdown-topics-description')
        ])
    ])
])


@app.callback(
    Output('markdown-article', 'children'),
    [Input('button-update', 'n_clicks')],
    [State('input-url', 'value')]
)
def update_article_description(n_clicks, url):
    if not n_clicks:
        return "No article queried yet."

    url = url.replace('.pdf', '')
    article_id = url.split('/')[-1]

    article = df[df.id == article_id]

    if article.shape[0] == 0:
        return "Unable to find article."

    article = article.iloc[-1]

    authors = ", ".join(eval(article.author))
    title = article.title.replace('\n  ', ' ')

    return dedent(f"""
    ### {title}

    ID: {article.id}

    Author(s): {authors}
    """)


@app.callback(
    Output('graph-topics', 'figure'),
    [Input('button-update', 'n_clicks')],
    [State('input-url', 'value')]
)
def update_topics_graph(n_clicks, url):
    if not n_clicks:
        return go.Figure()

    url = url.replace('.pdf', '')
    article_id = url.split('/')[-1]

    article = df[df.id == article_id]

    if article.shape[0] == 0:
        return "Unable to find article."

    article = article.iloc[-1]

    vector = tf_vectorizer.transform([article.summary])
    mixture = np.squeeze(lda.transform(vector))

    mix_df = pd.DataFrame(mixture, columns=['score'])

    fig = px.bar(mix_df, y='score', height=400)

    fig.update_layout(autosize=True)

    return fig


@app.callback(
    Output('markdown-topics-description', 'children'),
    [Input('graph-topics', 'selectedData'),
     Input('graph-topics', 'clickData')])
def display_hover_data(selectedData, clickData):
    if not clickData and not selectedData:
        return ""

    if not selectedData:
        selectedData = clickData

    strings = ""

    for data in selectedData['points']:
        topic_idx = data['x']
        score = data['y']

        topic = lda.components_[topic_idx]
        top_words = get_top_words(topic)

        strings += f"""
        Topic #{topic_idx} ({score * 100: .2f}% match)

        Keywords: {', '.join(top_words)}

        """

    return dedent(strings)


@app.callback(
    Output('markdown-recommendations', 'children'),
    [Input('button-update', 'n_clicks')],
    [State('input-url', 'value')]
)
def update_recommendations(n_clicks, url):
    if not n_clicks:
        return ""

    url = url.replace('.pdf', '')
    article_id = url.split('/')[-1]

    article = df[df.id == article_id]

    if article.shape[0] == 0:
        return "Unable to find article."

    article = article.iloc[-1]

    A = tm
    idx = df[df.id == article.id].index[0]
    D = A[idx].reshape(1, -1)

    A_norm = np.expand_dims(np.linalg.norm(A, axis=1), axis=1)
    D_norm = np.linalg.norm(D)

    S = np.squeeze(D.dot(A.T) / (D_norm * A_norm.T))

    k = 10
    top_idx = S.argsort()[:-k-1:-1]
    scores = S[top_idx]
    sim_df = df.iloc[top_idx].copy()
    sim_df['score'] = scores

    strings = "### Recommendations"

    for i in sim_df.index:
        rec_article = sim_df.loc[i]
        title = rec_article.title.replace('\n  ', ' ')
        authors = ", ".join(eval(rec_article.author))

        if rec_article.id != article_id:
            strings += dedent(f"""
            **{title} ({rec_article.score * 100 :.2f}% match)**

            * ID: [{rec_article.id}](https://arxiv.org/abs/{rec_article.id})
            * Authors: {authors}

            """)
    
    return strings


if __name__ == '__main__':
    app.run_server(debug=True)
