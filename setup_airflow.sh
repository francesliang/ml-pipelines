
PROJECT_NAME="ml-pipelines"
AIRFLOW_DAGS_PATH=~/airflow/dags/$PROJECT_NAME
AIRFLOW_DATA_PATH=~/airflow/data/$PROJECT_NAME

# Create related directories in airflow home directory
echo "Create project directories in airflow home directory"
mkdir $AIRFLOW_DAGS_PATH
mkdir $AIRFLOW_DATA_PATH

# Copy files to airflow home directory
echo "Copy project files to airflow home directory"
cp -r pipelines.py pipeline_utils.py utils/ $AIRFLOW_DAGS_PATH
cp -r data/* $AIRFLOW_DATA_PATH


