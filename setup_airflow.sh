
PROJECT_NAME="ml-pipelines"
AIRFLOW_DAGS_PATH=~/airflow/dags/$PROJECT_NAME
AIRFLOW_DATA_PATH=~/airflow/data/$PROJECT_NAME
AIRFLOW_PROJ_PATH=~/airflow/$PROJECT_NAME

# Create directories in airflow home directory
mkdir ~/airflow/dags
mkdir ~/airflow/data

# Create project directories in airflow home directory
echo "Create project directories in airflow home directory"
mkdir $AIRFLOW_DAGS_PATH
mkdir $AIRFLOW_DATA_PATH
mkdir $AIRFLOW_PROJ_PATH
mkdir $AIRFLOW_PROJ_PATH/logs
mkdir $AIRFLOW_PROJ_PATH/metadata

# Copy files to airflow home directory
echo "Copy project files to airflow home directory"
cp -r pipelines.py pipeline_utils.py $AIRFLOW_DAGS_PATH
cp -r data/* $AIRFLOW_DATA_PATH

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES


