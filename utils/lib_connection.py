import os
import cntk as ct
import mxnet as mx
from mxnet.contrib.onnx.onnx2mx.import_model import import_model

def lib_conn(temp):
	model_path = os.path.join(os.path.dirname(__file__), '../lib')
	return{
		temp == 'AGR' : get_AGR(model_path),
		temp == "PR" : get_PR(model_path),
		temp == "ER" : get_ER(model_path),
	}[1]

def get_PR(model_path):
	ctx = mx.gpu(0)
	image_size = (112,112)
	sym, arg_params, aux_params = import_model(model_path+'/'+'lib-pm.44.to.bin')
	model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
	model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
	model.set_params(arg_params, aux_params)
	return model

def get_AGR(model_path):
	AR_model = ct.Function.load(model_path+'/'+'lib-am.43.tc.bin')
	AR_model = ct.relu(AR_model)
	GR_model = ct.Function.load(model_path+'/'+'lib-gm.43.tc.bin')
	GR_model = ct.relu(GR_model)
	return AR_model, GR_model

def get_ER(model_path):
	ER_model = ct.Function.load(model_path+'/'+'lib-em.43.to.bin', format=ct.ModelFormat.ONNX)
	return ER_model