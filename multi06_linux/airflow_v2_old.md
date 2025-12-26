# Airflow



### 1. install

```bash
sudo vim ~/.bashrc

# airflow
export AIRFLOW_HOME=~/airflow
export PATH=$PATH:/home/big/.local/bin

[esc] :wq!

source ~/.bashrc

pip install apache-airflow

airflow db init

airflow users create \
    --username root \
    --password 1234 \
    --firstname dongheon \
    --lastname lee \
    --role Admin \
    --email admin@admin.com

# terminal 1
airflow webserver --port 8080
# terminal 2
airflow scheduler
# localhost:8080

```



### 2. settings

```bash
# metastore mysql로 변경 위해 db 설정
sudo apt install mysql-server -y

sudo service mysql start
sudo systemctl enable mysql
sudo service mysql status

sudo mysql -u root

show databases;

use mysql;

alter user 'root'@'localhost' identified with mysql_native_password by '1234';
CREATE USER 'root'@'%' IDENTIFIED BY '1234';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'localhost' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
flush privileges;

exit

mysql -u root -p
exit


sudo apt install libmysqlclient-dev -y
sudo apt install pkg-config -y

pip install mysqlclient
pip install mysql-connector-python
pip install apache-airflow-providers-mysql

# mysql
mysql -u root -p

create database airflow character set utf8mb4 collate utf8mb4_unicode_ci;

create user 'airflow'@'localhost' identified by '1234';
create user 'airflow'@'%' identified by '1234';

grant all privileges on airflow.* to 'airflow'@'localhost';
grant all privileges on airflow.* to 'airflow'@'%';

flush privileges;

exit;


cd airflow
vim airflow.cfg

:36
default_timezone = Asia/Seoul

:45
executor = LocalExecutor

:106
load_examples = False

:424
sql_alchemy_conn =  mysql://airflow:1234@localhost:3306/airflow

:999
endpoint_url = http://localhost:8988

:1171
base_url = http://localhost:8988

:1181
default_ui_timezone = Asia/Seoul

:1193
web_server_port = 8988

:1797
dag_dir_list_interval = 60

[esc] :wq!

airflow db init

airflow users create \
    --username admin \
    --password 1234 \
    --firstname dongheon \
    --lastname lee \
    --role Admin \
    --email admin@admin.com

mkdir -p ~/airflow/dags

airflow scheduler
airflow webserver

# nohup airflow webserver &
# nohup airflow scheduler &
# localhost:8988 접속 -> admin / 1234

```
