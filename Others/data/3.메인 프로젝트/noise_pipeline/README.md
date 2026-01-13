# 도시소음 데이터를 활용한 소음 개선 정보 제공 서비스 개발

> 도시 소음 공공 데이터를 수집·분석하여
> 소음 현황 시각화, 개선 필요 지역 안내, 신규 위치 소음 개선 예측 기능을 제공하는 데이터 파이프라인 서비스



### 프로젝트 구조

```bash
# /home/big/noise_pipeline
~/noise_pipeline/
├── config/
│   └── settings.py
├── dags/
│   └── noise_pipeline_dag_ml.py
├── data/
│   ├── raw/
│   │   └── gangnam_noise_data.csv
│   └── processed/
│       ├── result.csv
│       └── model.pkl
├── spark/
│   └── noise_analysis.py
├── src/
│   ├── fetcher.py
│   ├── kafka_producer.py
│   ├── main.py
│   ├── processor.py
│   └── ml.py
├── streamlit/
│   └── app.py
├── requirements.txt
└── README.md
```



### 전체 아키텍처
[공공 소음 API]
        ↓
[Airflow DAG]
 ├─ 1. 데이터 수집 (PythonOperator)
 ├─ 2. Kafka 전송 (PythonOperator)
 ├─ 3. PySpark 분석 (SparkSubmitOperator)
 └─ 4. ML 모델 학습 (PythonOperator)
        ↓
[data/processed/result.csv + model.pkl]
        ↓
[Streamlit UI]
 ├─ 지도 시각화
 ├─ 상세 테이블
 └─ 신규 위치 소음 개선 필요 예측



### 컨테이너 생성

```bash
docker run -it --name noise-dev -p 8080:8080 -p 8501:8501 -p 9092:9092 -d ubuntu:24.04

# 포트 설명
# 8080 : Airflow
# 8501 : Streamlit
# 9092 : Kafka
```



### 기본 설치

```bash
docker exec -it noise-dev /bin/bash

apt update
apt upgrade -y

# 계정설정
useradd -m -s /bin/bash big
passwd big
# 1234

apt install sudo -y
usermod -aG sudo big

su big
cd

# 기본 패키지 설치
sudo apt install vim -y
# 5 (Asia), 68 (Seoul)
sudo apt install python3 -y
sudo apt install python3-pip -y

sudo apt install wget -y
wget https://corretto.aws/downloads/latest/amazon-corretto-17-x64-linux-jdk.tar.gz
tar -xvzf amazon-corretto-17-x64-linux-jdk.tar.gz
ln -s amazon-corretto-17.0.17.10.1-linux-x64 java

# miniconda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# enter, yes 등 입력하여 설치 완료

exit
su big

conda config --set auto_activate_base false
exit
su big
cd

# 설정
sudo vim ~/.bashrc

# python alais
alias python=python3

# java home
export JAVA_HOME=/home/big/java
export PATH=$PATH:$JAVA_HOME/bin

[esc] :wq!

source ~/.bashrc

# 확인
java -version
javac -version
python -V
pip -V

conda create -n noise python
conda activate noise

rm Miniconda3-latest-Linux-x86_64.sh 

# requirements.txt로 library 설치 방법
# pip install -r requirements.txt
```



### pipeline 설치

```bash
# kafka 설치
wget https://dlcdn.apache.org/kafka/4.1.1/kafka_2.13-4.1.1.tgz
tar -xvzf kafka_2.13-4.1.1.tgz
ln -s kafka_2.13-4.1.1 kafka

sudo vim ~/.bashrc

# kafka
export KAFKA_HOME=/home/big/kafka
export PATH=$PATH:$KAFKA_HOME/bin

[esc] :wq!

source ~/.bashrc

cd kafka
KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"

bin/kafka-storage.sh format --standalone -t $KAFKA_CLUSTER_ID -c config/server.properties

sudo vim ~/.bashrc

# kafka 시작/종료 alias
alias kafka_start='~/kafka/bin/kafka-server-start.sh -daemon ~/kafka/config/server.properties'
alias kafka_stop='~/kafka/bin/kafka-server-stop.sh'

[esc] :wq!

source ~/.bashrc

kafka_start

$KAFKA_HOME/bin/kafka-topics.sh --create --topic noise_raw_data --bootstrap-server localhost:9092


# spark 설치
wget https://dlcdn.apache.org/spark/spark-4.1.1/spark-4.1.1-bin-hadoop3.tgz
tar -xvzf spark-4.1.1-bin-hadoop3.tgz
ln -s spark-4.1.1-bin-hadoop3 spark

sudo vim ~/.bashrc

# spark
export SPARK_HOME=/home/big/spark 
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

source ~/.bashrc


cd $SPARK_HOME/conf

cp spark-env.sh.template spark-env.sh
vim spark-env.sh

export JAVA_HOME=/home/big/java
export PYSPARK_PYTHON=/home/big/miniconda3/envs/noise/bin/python
export PYSPARK_DRIVER_PYTHON=/home/big/miniconda3/envs/noise/bin/python

[esc] :wq!

cp spark-defaults.conf.template spark-defaults.conf
vim spark-defaults.conf

spark.executorEnv.JAVA_HOME 		/home/big/java

[esc] :wq!


# airflow

sudo vim ~/.bashrc

# airflow
export AIRFLOW_HOME=/home/big/airflow
export PATH=$PATH:/home/big/.local/bin

# Airflow 일괄 실행
alias airflow_start='airflow api-server -D && airflow scheduler -D && airflow dag-processor -D'
# Airflow 일괄 종료
alias airflow_stop="ps -ef | grep 'airflow' | grep -v grep | awk '{print \$2}' | xargs -r kill -9"

source ~/.bashrc

conda activate noise

pip install apache-airflow

airflow db migrate
cd airflow
vim airflow.cfg

# line 43
default_timezone = Asia/Seoul

# line 150
load_examples = False

# line 2188
refresh_interval = 60

[esc] :wq!

rm airflow.db
airflow db migrate

vim $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated

{"admin": "1234"}

[esc] :Wq!


pip install pyspark
pip install grpcio-status
pip install apache-airflow-providers-apache-spark


# library
pip install kafka-python
pip install streamlit
pip install scikit-learn
```





### 프로젝트 복사

```bash
# host 컴퓨터에서
docker cp noise_pipeline noise-dev:/home/big/noise_pipeline/

# docker 내부에서
sudo vim ~/.bashrc

# 가상환경 python path 설정
export PYTHONPATH=$PYTHONPATH:/home/big/noise_pipeline
[esc]:wq!

source ~/.bashrc

conda activate noise

cd ~/airflow
vim airflow.cfg

# line 7
dags_folder = /home/big/noise_pipeline/dags/

[esc] :wq!

rm airflow.db
airflow db migrate


sudo chown big:big -R noise_pipeline

cd ~/noise_pipeline

touch src/__init__.py
touch config/__init__.py
touch dags/__init__.py
touch spark/__init__.py

```





### 수동 실행

```bash
python src/main.py
# ls data/raw

spark-submit spark/noise_analysis.py
# ls data/processed

python src/ml.py
# ls data/processed/

streamlit run streamlit/app.py
# localhost:8501
```

