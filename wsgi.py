from views import views

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Machine_Learning'
    app.secret_key = 'Machine_Learning'

    app.register_blueprint(views, url_prefix='/')
    return app

app = create_app()

if (name == '__main__'):
    app.run()