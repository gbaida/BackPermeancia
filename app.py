from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
#from flask_cors import CORS
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np

flask_app = Flask(__name__)
#cors = CORS(flask_app)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "ML React App", 
		  description = "Predição de Permeância")


name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  { 'PP_ALTURA_UTIL': fields.Float(required = True),
					'PP_FACE': fields.Float(required = True),
					'PP_FUNDO': fields.Float(required = True),
					'PP_FUNDO_INF': fields.Float(required = True),
					'PP_VALVULA': fields.Float(required = True),
					'PP_FUROS_COLADEIRA': fields.Float(required = True),
					'PP_DIAMETRO_FUROS_COLADEI': fields.Float(required = True),
					'PP_NUMERO_FOLHAS': fields.Float(required = True),
					'PP_GRAMATURA_EXTERNO': fields.Float(required = True),
					'PP_GRAMATURA_2': fields.Float(required = True),
					'PP_GRAMATURA_3': fields.Float(required = True),
					'PP_GRAMATURA_4': fields.Float(required = True),
					'PP_GRAMATURA_5': fields.Float(required = True),
					'PP_GRAMATURA_INTERNO': fields.Float(required = True),
					'PP_LARGURA_PEAD': fields.Float(required = True),
					'PP_TIPO_SACO_COLADO': fields.String(required = True),
					'PP_PATCH': fields.String(required = True),
					'PP_PERFURADO_TUBEIRA': fields.String(required = True),
					'PP_PERFURADO_COLADEIRA': fields.String(required = True),
					'PP_PAPEL_EXTERNO': fields.String(required = True),
					'PP_PAPEL_2': fields.String(required = True),
					'PP_PAPEL_3': fields.String(required = True),
					'PP_PAPEL_4': fields.String(required = True),
					'PP_PAPEL_5': fields.String(required = True),
					'PP_PAPEL_INTERNO': fields.String(required = True),
					'PP_TIP_PERF_FOL_EXT': fields.String(required = True),
					'PP_TIP_PERF_FOL_2': fields.String(required = True),
					'PP_TIP_PERF_FOL_3': fields.String(required = True),
					'PP_TIP_PERF_FOL_4': fields.String(required = True),
					'PP_TIP_PERF_FOL_5': fields.String(required = True),
					'PP_TIP_PERF_FOL_INT': fields.String(required = True),
					'PP_VOLUME': fields.Float(required = True),
					})

def predicao(data):
	filename = 'modeloXGBPermeancia.pkl'
	with open(filename, 'rb') as file:
		model = pickle.load(file)

	df2 = pd.read_csv('Dados.csv',nrows=1)
	df2.drop(columns=['PP_PERMEANCIA'], inplace=True)

	for column in df2.columns:
		df2[column] =  float(0)

	df1 = pd.DataFrame(data,index=[0])

	for column in df1.columns:
		if (df1[column].all() == '0'):
			df1[column] = pd.to_numeric(df1[column])

	categorical_feature_mask = df1.dtypes==object
	categorical_cols = df1.columns[categorical_feature_mask].tolist()

	for column in df1[categorical_cols]:
		df1[column] = [str(value) for value in df1[column].values]

	for column in df1[categorical_cols]:
		dum_df1 = pd.get_dummies(df1[column], columns=[column], prefix=column)
		df1 =df1.join(dum_df1)
	
	df1.drop(columns = categorical_cols, inplace=True)

	for column in df1.columns:
		if column in df2.columns:
			df2[column] = df1[column]

	dtest = xgb.DMatrix(df2)
	return np.round(model.predict(dtest)[0],2)

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "Origin, Content-type, Accept")
		response.headers.add('Access-Control-Allow-Methods', "GET, POST, OPTIONS")
		return response

	@app.expect(model)		
	def post(self):
		try: 
			formData = request.json
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": str(predicao(formData))
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		
			
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
	
		
