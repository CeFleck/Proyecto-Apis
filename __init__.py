from flask import Flask
from flask_restful import Resource, Api
import proy_api


app=Flask(__name__)
api=Api(app)
api.add_resource(proy_api.graficas,'/')
api.add_resource(proy_api.lin,'/lin')
api.add_resource(proy_api.ran,'/ran')
api.add_resource(proy_api.rbf,'/rbf')


if __name__=='__main__':
  app.run(debug=True)