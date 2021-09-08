#!/usr/bin/env python
# coding: utf-8

import joblib
import pickle
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


# --------------------------------------------------------------
# Funciones para la creación del dataset a partir de
# la ventana de datos originales
# --------------------------------------------------------------

def crea_columnas_pasadas(atributo, datos, ventana=1):
    serie = pd.Series(datos[atributo])
    if ventana > len(serie) - 1:
        print('No es posible generar un dataset de ventana {} con una serie de tamaño {}'.format(ventana, len(serie)))
        return None
    columnas = ['{}-{}'.format(atributo, i + 1) for i in range(ventana)]
    dataset = pd.DataFrame(columns=columnas)
    for i in range(ventana):
        serie_desplazada = serie.shift(i + 1)
        dataset[columnas[i]] = serie_desplazada

    dataset.dropna(inplace=True)

    dataset = dataset[reversed(columnas)].copy()
    dataset[atributo] = serie.iloc[ventana:]
    return dataset


def crea_ventana_atributos(datos, atributos, ventana=1):
    resultado = pd.DataFrame()
    for a in datos.columns:
        resultado = pd.concat([crea_columnas_pasadas(a, datos, ventana), resultado], axis=1)
    return resultado


def calcula_tipo_scheduler(scheduled_by):
    if 'Mesos' in scheduled_by:
        return 'Mesos'
    elif 'Omega' in scheduled_by:
        return 'Omega'
    else:
        print('Scheduler desconocido')
        return 'Desconocido'


def calcula_previous_task_scheduled(fila):
    if fila['previous_submission_time'] + fila['previous_queue_time_till_first_scheduled'] < fila['submission_time']:
        return True
    else:
        return False


def calcula_previous_task_time_scheduled(fila):
    if fila['previous_submission_time'] + fila['previous_queue_time_till_first_scheduled'] < fila['submission_time']:
        return fila['previous_queue_time_till_first_scheduled']
    else:
        return fila['submission_time'] - fila['previous_submission_time']


def calcula_previous_task_finished(fila):
    if fila['previous_submission_time'] + fila['previous_queue_time_till_fully_scheduled'] < fila['submission_time']:
        return True
    else:
        return False


def calcula_previous_task_time_finished(fila):
    if fila['previous_submission_time'] + fila['previous_queue_time_till_fully_scheduled'] < fila['submission_time']:
        return fila['previous_queue_time_till_fully_scheduled']
    else:
        return fila['submission_time'] - fila['previous_submission_time']


def calcula_vector_atributos(datos, atributos, ventana):
    # Cálculo de 'previous_task_finished' y 'previous_task_time_finished'
    #datos['previous_submission_time'] = datos['submission_time'].shift(1).fillna(0)
    #Viene del simulador
    #datos['previous_queue_time_till_fully_scheduled'] = datos['queue_time_till_fully_scheduled'].shift(1).fillna(0)
    #Viene del simulador
    datos['previous_task_finished'] = datos['finished'].shift(1).fillna(0)
    #datos['previous_task_finished'] = datos.apply(calcula_previous_task_finished, axis=1)
    datos['previous_task_finished'] = pd.Categorical(datos['previous_task_finished'], categories=[True, False])
    #Viene del simulador
    #datos['previous_task_time_finished'] = datos.apply(calcula_previous_task_time_finished, axis=1)
    datos['previous_task_time_finished'] = datos['queue_time_till_fully_scheduled'].shift(1).fillna(0)
    # Cálculo de 'previous_task_scheduled' y 'previous_task_time_scheduled'
    #Viene del simulador
    #datos['previous_queue_time_till_first_scheduled'] = datos['queue_time_till_first_scheduled'].shift(1).fillna(0)
    #Viene del simulador
    datos['previous_task_scheduled'] = datos['scheduled'].shift(1).fillna(0)
    #datos['previous_task_scheduled'] = datos.apply(calcula_previous_task_scheduled, axis=1)
    datos['previous_task_scheduled'] = pd.Categorical(datos['previous_task_scheduled'], categories=[True, False])
    #datos['previous_task_time_scheduled'] = datos.apply(calcula_previous_task_time_scheduled, axis=1)
    #Viene del simulador
    datos['previous_task_time_scheduled'] = datos['queue_time_till_first_scheduled'].shift(1).fillna(0)

    # Cálculo de 'submission_time_delta'
    datos['submission_time_delta'] = datos['submission_time'].diff().fillna(0)
    datos['submission_time_delta'] = datos['submission_time_delta'].apply(lambda x: 0 if x < 0 else x)

    # Cálculo de atributos logarítmicos:
    #   - 'log_queue_time_till_fully_scheduled'
    #   - 'log_submission_time_delta'
    #   - 'log_num_tasks'

    # tiempo_minimo = datos[datos['queue_time_till_fully_scheduled']>0]['queue_time_till_fully_scheduled'].min()
    # tiempo_minimo = 0.0001
    # datos['log_queue_time_till_fully_scheduled'] = datos['queue_time_till_fully_scheduled'].apply(lambda x: tiempo_minimo if x==0 else x)
    # datos['log_queue_time_till_fully_scheduled'] = np.log(datos['log_queue_time_till_fully_scheduled'])

    # duration_minimo = datos[datos['estimated_task_duration']>0]['estimated_task_duration'].min()
    duration_minimo = 0.0001
    datos['log_estimated_task_duration'] = datos['estimated_task_duration'].apply(
        lambda x: duration_minimo if x == 0 else x)
    datos['log_estimated_task_duration'] = np.log(datos['log_estimated_task_duration'])

    # delta_minimo = datos[datos['submission_time_delta']>0]['submission_time_delta'].min()
    delta_minimo = 0.0001
    datos['log_submission_time_delta'] = datos['submission_time_delta'].apply(lambda x: delta_minimo if x == 0 else x)
    datos['log_submission_time_delta'] = np.log(datos['log_submission_time_delta'])

    datos['log_num_tasks'] = np.log(datos['num_tasks'])

    # Cálculo de atributos categóricos
    datos['scheduler_type'] = datos['scheduled_by'].apply(calcula_tipo_scheduler)
    datos['scheduler_type'] = pd.Categorical(datos['scheduler_type'], categories=['Mesos', 'Omega'])
    datos['workload_type'] = pd.Categorical(datos['workload_type'], categories=['Batch', 'Service'])

    # Cálculo de dataset filtrado
    datos = datos[atributos].copy()
    datos = pd.get_dummies(datos)

    if ventana > 0:
        X = crea_ventana_atributos(datos, atributos, ventana)
    else:
        X = datos

    return X.iloc[-1]


# --------------------------------------------------------------
# Clase Estimador
# --------------------------------------------------------------
class EstimadorQueueFinished():
    def __init__(self, modelo_mesos, modelo_omega, atributos, ventana, tipo_modelo='joblib'):
        self.atributos = atributos
        self.ventana = ventana
        if 'log' in modelo_mesos and 'log' in modelo_omega:
            self.log = True
        elif 'log' not in modelo_mesos and 'log' not in modelo_omega:
            self.log = False
        else:
            print("No se pueden mezclar modelos 'log' y 'no log'")
        if tipo_modelo == 'pickle':
            self.regresor_mesos = pickle.load(open('./models/{}.pickle'.format(modelo_mesos), 'rb'))
            self.regresor_omega = pickle.load(open('./models/{}.pickle'.format(modelo_omega), 'rb'))
        elif tipo_modelo == 'joblib':
            self.regresor_mesos = joblib.load('./models/{}.joblib'.format(modelo_mesos))
            self.regresor_omega = joblib.load('./models/{}.joblib'.format(modelo_omega))

    def predict(self, ventana_de_datos):
        vector = calcula_vector_atributos(ventana_de_datos, self.atributos, self.ventana)
        estimacion_mesos = self.regresor_mesos.predict([vector])[0]
        estimacion_omega = self.regresor_omega.predict([vector])[0]
        if self.log:
            estimacion_mesos = np.exp(estimacion_mesos)
            estimacion_omega = np.exp(estimacion_omega)
        return {'estimacion_mesos': estimacion_mesos, 'estimacion_omega': estimacion_omega}



