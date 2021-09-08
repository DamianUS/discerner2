# load Flask
import flask
import pandas as pd
from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf
import joblib
from config import *
import time
import json
import logging
from estimador_finished_queue import EstimadorQueueFinished
from estimador_no_finished_no_queue import EstimadorNoFinishedNoQueue
from estimador_finished_no_queue import EstimadorFinishedNoQueue
from estimador_scheduling_time import EstimadorSchedulingTime
app = flask.Flask("jose_predict")

pd.set_option('display.max_columns', None)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
ventana = 10
estrategias = ["no_finished_no_queue", "finished_no_queue", "finished_queue", "scheduling_time"]
#Seleccion estrategia
estrategia = estrategias[3]
if estrategia == "no_finished_no_queue":
    atributos = ['estimated_task_duration', 'data_center_utilization_at_submission', 'num_tasks',
                 'submission_time_delta', 'scheduler_type', 'workload_type']
    estimador = EstimadorNoFinishedNoQueue("log_mesos_W05IA85", "log_omega_W05IA85", atributos, ventana)
elif estrategia == "finished_no_queue":
    atributos = ['estimated_task_duration', 'data_center_utilization_at_submission', 'num_tasks',
                 'submission_time_delta', 'scheduler_type', 'workload_type', 'previous_task_finished']
    estimador = EstimadorFinishedNoQueue("log_mesos_W05IA85_task_finished", "log_omega_W05IA85_task_finished", atributos, ventana)
elif estrategia == "finished_queue":
    atributos = ['estimated_task_duration', 'data_center_utilization_at_submission', 'num_tasks',
             'submission_time_delta', 'scheduler_type', 'workload_type',
             'previous_task_finished', 'previous_task_time_finished',
             'previous_task_scheduled', 'previous_task_time_scheduled']
    estimador = EstimadorQueueFinished("log_mesos_W05IA85_previous_task_finished_scheduled", "log_omega_W05IA85_previous_task_finished_scheduled", atributos, ventana)
else:
    #LO QUE QUIERO
    atributos = ['estimated_task_duration', 'data_center_utilization_at_submission', 'num_tasks',
                 'job_scheduling_attempts', 'tasks_scheduling_attempts',
                 'submission_time_delta', 'scheduler_type', 'workload_type',
                 'previous_task_finished', 'previous_task_time_finished',
                 'previous_task_scheduled', 'previous_task_time_scheduled',
                 'previous_task_scheduling_time', 'previous_task_scheduling_time_finished']
    estimador = EstimadorSchedulingTime("log_mesos_W05IA100_clase_scheduling_time_e3", "log_omega_W05IA100_clase_scheduling_time_e3", atributos, ventana)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    parameters = [json.loads(row) for row in flask.request.form.values()]
    #LO QUE LE PASO
    columns = ['estimated_task_duration', 'data_center_utilization_at_submission', 'num_tasks',
               'job_scheduling_attempts', 'tasks_scheduling_attempts',
     'submission_time', 'scheduled_by', 'workload_type', 'finished', 'queue_time_till_fully_scheduled',
               'scheduled', 'queue_time_till_first_scheduled', 'scheduling_time']
    df = pd.DataFrame(parameters, columns=columns)
    df['scheduled_by'] = 'Mesos'
    #print(df)
    #start_time = time.time()
    res_dict = estimador.predict(df[:11])
    #print("--- %s seconds ---" % (time.time() - start_time))
    #print("Inter-arrival:", df['submission_time'].diff().fillna(0).mean(), "Mesos:", res_dict['estimacion_mesos'], "Omega:", res_dict['estimacion_omega'])
    return "Mesos" if res_dict['estimacion_mesos'] < res_dict['estimacion_omega'] else "Omega"


#
# # define a predict function as an endpoint
# @app.route("/predict-mesos", methods=["GET", "POST"])
# def predict_mesos():
#     data = {"success": False}
#     params = flask.request.query_string
#     array_parametros = [float(s) for s in params.decode("utf-8").split(",")]
#     prediction = predict_mesos_internal(array_parametros)
#     return flask.jsonify(prediction)
#
#
# def predict_mesos_internal(array_parametros):
#     prepared_array = scaler.transform(np.array(array_parametros))
#     #prepared_array = np.array(array_parametros).reshape(1, -1) #ndArray(1,6)
#     prediction_input_array = np.array([prepared_array]) #ndArray(1,1,6)
#     prediction_ndarray = model_mesos.predict(prediction_input_array);
#     prediction_ndarray = prediction_ndarray / scaler.scale_[PREDICTED_COLUMN_INDEX];
#     prediction = prediction_ndarray.tolist()[0][0]
#
#     return prediction
#
# @app.route("/predict-omega", methods=["GET", "POST"])
# def predict_omega():
#     data = {"success": False}
#
#     params = flask.request.query_string
#     array_parametros = [float(s) for s in params.decode("utf-8").split(",")]
#     prediction = predict_omega_internal(array_parametros);
#
#     return flask.jsonify(prediction)
#
# def predict_omega_internal(array_parametros):
#     prepared_array = scaler.transform(np.array(array_parametros))
#     #prepared_array = np.array(array_parametros).reshape(1, -1)  # ndArray(1,6)
#     prediction_input_array = np.array([prepared_array])  # ndArray(1,1,6)
#     prediction_ndarray = model_omega.predict(prediction_input_array)
#     prediction_ndarray = prediction_ndarray / scaler.scale_[PREDICTED_COLUMN_INDEX]
#     prediction = prediction_ndarray.tolist()[0][0]
#
#     return prediction



# start the flask app, allow remote connections
app.run(host='0.0.0.0')
