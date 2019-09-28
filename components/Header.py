import dash_html_components as html
from .Column import Column


def Header(title, app):
    height = 60
    return html.Div(
        style={
            'borderBottom': 'thin lightgrey solid',
            'marginRight': 20,
            'marginBottom': 20,
            'height': height
        },
        children=[
            Column(
                width=6,
                children=title,
                style={
                    'fontSize': 35,
                }
            ),
            Column(
                width=6,
                children=html.Img(
                    src=app.get_asset_url('images/dash-logo.png'),
                    style={
                        'float': 'right',
                        'height': height,
                    }
                )
            )
        ]
    )
