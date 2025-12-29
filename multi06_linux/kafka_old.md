# Apache Kafka





```bash
# java 11 사용
wget https://corretto.aws/downloads/latest/amazon-corretto-11-x64-linux-jdk.tar.gz
tar xvzf amazon-corretto-11-x64-linux-jdk.tar.gz
ln -s amazon-corretto-11.0.20.8.1-linux-x64/ java

# java path
sudo vim ~/.bashrc

# java
export JAVA_HOME=/home/big/java
export PATH=$PATH:$JAVA_HOME/bin
[esc] :wq!

source ~/.bashrc
java -version
javac -version

# zookeeper 다운로드
# apache zookeeper -> download -> 3.9.4(최신버전)
wget https://dlcdn.apache.org/zookeeper/zookeeper-3.9.4/apache-zookeeper-3.9.4-bin.tar.gz

tar xvzf apache-zookeeper-3.9.4-bin.tar.gz


# java 11버전이 제대로 지원되는 버전 (3.9.0) -> 이후 버전은 java version을 올려야 함
wget https://dlcdn.apache.org/kafka/3.9.0/kafka_2.13-3.9.0.tgz

tar xvzf kafka_2.13-3.9.0.tgz
ln -s kafka_2.13-3.9.0 kafka

cd kafka

KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"

```

`vim config/server.properties`

```properties
# 주석 해제 (34 line)
listener=PAINTEXT://:9092
```

```bash
# server 실행 (zookeeper 같이 실행)
bin/zookeeper-server-start.sh config/zookeeper.properties

# 새 창 열어서 broker 실행
cd kafka
bin/kafka-server-start.sh config/server.properties

# 새 창 열어서 topic 생성
cd kafka
bin/kafka-topics.sh --create --topic multi --bootstrap-server localhost:9092

# 생성된 topic 목록 확인
bin/kafka-topics.sh --list --bootstrap-server localhost:9092

# multi 확인
bin/kafka-topics.sh --describe --topic multi --bootstrap-server localhost:9092

# producer 실행하여 topic에 이벤트 등록
bin/kafka-console-producer.sh --topic multi --bootstrap-server localhost:9092
> multi
> kafka

# 새 창 열어서 consumer 실행하여 이벤트 확인
cd kafka
bin/kafka-console-consumer.sh --topic multi --from-beginning --bootstrap-server localhost:9092
```



