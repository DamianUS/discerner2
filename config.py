ROOT_DIR = '/Users/damianfernandez/PycharmProjects/discerner2/'
DATA = ROOT_DIR + "data/dataset.csv"
MESOS_DATA = ROOT_DIR + "data/mesos.csv"
OMEGA_DATA = ROOT_DIR + "data/omega.csv"
SCALER = ROOT_DIR + "scalers/quantile_scaler.save"
MESOS_MODEL = ROOT_DIR + "models/mesos"
OMEGA_MODEL = ROOT_DIR + "models/omega"
BATCH_SIZE = 30
EPOCHS = 10
ANN_BATCH_SIZE = 32
UNTAIL = False
UNTAIL_MAX_PERCENTILE = 0.6
UNTAIL_MIN_PERCENTILE = 0.0
UNTAIL_MODE = "remove"  # or "collapse"
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.2

COLUMNS = [
    'submission_time',
    'num_tasks',
    'cpus_per_task',
    'mem_per_task',
    'estimated_task_duration',
    'queue_time_till_fully_scheduled',
    'makespan',
    'data_center_utilization_at_submission'
]
ADDITIONAL_COLUMNS = {
    "cpus_per_batch": "cpus_per_task",
    "mem_per_batch": "mem_per_task"
}
# CPU y MEM per batch
# Num_tasks
SELECTED_COLUMNS = [
    'estimated_task_duration',
    'data_center_utilization_at_submission',
    'num_tasks',
    # 'cpus_per_batch',
    # 'mem_per_batch',
    #'submission_time_transformed',
    'submission_time',
    'queue_time_till_fully_scheduled'
]
# CPU y MEM per batch = 5
# Num_tasks = 4
PREDICTED_COLUMN_INDEX = 4
PREDICTED_COLUMN_NAME = 'queue_time_till_fully_scheduled'
