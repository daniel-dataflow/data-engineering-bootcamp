# ElasticStack



```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.12-linux-x86_64.tar.gz

wget https://artifacts.elastic.co/downloads/logstash/logstash-7.17.12-linux-x86_64.tar.gz

wget https://artifacts.elastic.co/downloads/kibana/kibana-7.17.12-linux-x86_64.tar.gz

tar xvzf elasticsearch-7.17.12-linux-x86_64.tar.gz
tar xvzf logstash-7.17.12-linux-x86_64.tar.gz
tar xvzf kibana-7.17.12-linux-x86_64.tar.gz

ln -s elasticsearch-7.17.12 elastic
ln -s logstash-7.17.12 logstash
ln -s kibana-7.17.12-linux-x86_64 kibana

sudo vim ~/.bashrc

# elk
export ELASTIC_HOME=/home/big/elastic
export LOGSTASH_HOME=/home/big/logstash
export KIBANA_HOME=/home/big/kibana
export PATH=$PATH:$ELASTIC_HOME/bin:$LOGSTASH_HOME/bin:$KIBANA_HOME/bin

[esc] :wq!

source ~/.bashrc

logstash-plugin install logstash-integration-jdbc

vim ~/logstash/test.conf

```

*vim ~/logstash/test.conf*

```bash
input {
    jdbc {
        jdbc_driver_library => "/usr/share/java/mysql-connector-j-8.1.0.jar"
        jdbc_driver_class => "com.mysql.cj.jdbc.Driver"
        jdbc_connection_string => "jdbc:mysql://localhost:3306/mysql"
        jdbc_user => "root"
        jdbc_password => "1234"
        statement => "SELECT * from test"
        schedule => "* * * * *"
    }
}
filter {
}
output {
    elasticsearch {
        hosts => ["localhost:9200"]
        index => "test"
        document_id => "%{id}"
    }
}

```

```bash
elasticsearch -d
nohup kibana &
nohup logstash -f ~/logstash/test.conf &
# 비밀번호 설정 없음!
```



```bash
# 추가설정
# max file descriptors
# <domain> <type> <item> <value>
sudo vim /etc/security/limits.conf
big - nofile 65535

[esc] :wq!

# max virtual memory areas
sudo vim /etc/sysctl.conf
vm.max_map_count=262144

[esc] :wq!

```




