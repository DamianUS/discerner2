import joblib
import pickle

from sklearn2pmml import make_pmml_pipeline, PMMLPipeline, sklearn2pmml

mesos = pickle.load(open('./models/{}.pickle'.format('log_mesos_W05IA100_clase_scheduling_time_e3'), 'rb'))
omega = pickle.load(open('./models/{}.pickle'.format('log_omega_W05IA100_clase_scheduling_time_e3'), 'rb'))

#regresor_mesos = joblib.load('./models/{}.pickle'.format('log_mesos_W05IA100_clase_scheduling_time_e3'))
#regresor_omega = joblib.load('./models/{}.pickle'.format('log_omega_W05IA100_clase_scheduling_time_e3'))


mesos_pipeline = make_pmml_pipeline(mesos)
omega_pipeline = make_pmml_pipeline(omega)

sklearn2pmml(mesos_pipeline, "models/log_mesos_W05IA100_clase_scheduling_time_e3.pmml")
sklearn2pmml(omega_pipeline, "models/log_omega_W05IA100_clase_scheduling_time_e3.pmml")