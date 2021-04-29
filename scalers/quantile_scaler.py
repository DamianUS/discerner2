import joblib
import config
import utils
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd

SCALER_NAME = "quantile_scaler"

dataframe = utils.prepare_dataframe()

dataframe[config.SELECTED_COLUMNS].hist()
plt.title("Histogram for raw data")
plt.show()
# SCALER GENERATION
scaler = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
X, Y = utils.prepare_data_for_model(dataframe, config.SELECTED_COLUMNS, scaler)

# TEST SCALER
transformed = scaler.transform(dataframe[config.SELECTED_COLUMNS])
# CPU y MEM per batch transformed_dataset = pd.DataFrame({'estimated_task_duration': transformed[:, 0],
# 'data_center_utilization_at_submission': transformed[:, 1], 'cpus_per_batch': transformed[:, 2], 'mem_per_batch':
# transformed[:, 3], 'submission_time_transformed': transformed[:, 4], 'queue_time_till_fully_scheduled':
# transformed[:,5]}) Num_tasks
transformed_dataset = pd.DataFrame(
    {'estimated_task_duration': transformed[:, 0], 'data_center_utilization_at_submission': transformed[:, 1],
     'num_tasks': transformed[:, 2], 'submission_time_transformed': transformed[:, 3],
     'queue_time_till_fully_scheduled': transformed[:, 4]})
transformed_dataset.hist()
plt.title("Histogram for transformed values")
plt.show()

# SAVE SCALER
scaler_filename = "./" + SCALER_NAME + ".save"
joblib.dump(scaler, scaler_filename)
print("terminado")
