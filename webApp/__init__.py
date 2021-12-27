from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Machine_Learning'
    app.secret_key = 'Machine_Learning'

    from webApp.views import views

    app.register_blueprint(views, url_prefix='/')
    return app
