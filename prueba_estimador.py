#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from estimador_queue_time import Estimador


# Configuración
atributos = ['estimated_task_duration', 'data_center_utilization_at_submission', 'num_tasks', 
             'submission_time_delta', 'scheduler_type', 'workload_type',
             'previous_task_finished', 'previous_task_time_finished',
             'previous_task_scheduled', 'previous_task_time_scheduled']
ventana = 10



# Preparación de la ventana de datos previos
ventana_de_prueba = pd.read_csv('./data/completo_W05IA85.csv', sep=' ')[:11] # Al menos una fila más que el tamaño de la ventana

# Creación del estimador
estimador = Estimador('log_mesos_W05IA85_previous_task_finished_scheduled', 'log_omega_W05IA85_previous_task_finished_scheduled', atributos, ventana)


# Predicción de tiempo en cola según scheduler
print(estimador.predict(ventana_de_prueba))




