```bash
// 분산 저장하는 설정을 방법

mkdir \mongo\config, \mongo\shard1, \mongo\shard2

mongod --configsvr --dbpath C:\\mongo\config --port 27020 --replSet rs_config

# 새 창에서
mongosh --port 27020
rs.initiate({
  _id: "rs_config",
  configsvr: true,
  members: [
    { _id: 0, host: "localhost:27020" }
  ]
});
exit;

# 새 창에서
mongod --shardsvr --dbpath C:\mongo\shard1 --port 27021 --replSet rs_shard1 
# 새 창에서
mongosh --port 27021
rs.initiate({ _id: "rs_shard1", members: [{ _id: 0, host: "localhost:27021" }] 
exit;

# 새 창에서
mongod --shardsvr --dbpath C:\mongo\shard2 --port 27022 --replSet rs_shard2
# 새 창에서
mongosh --port 27022
rs.initiate({ _id: "rs_shard2", members: [{ _id: 0, host: "localhost:27022" }] });
exit;

# 새 창에서
mongos --configdb rs_config/localhost:27020 --port 27023

# 새 창에서
mongosh --port 27023
sh.addShard("rs_shard1/localhost:27021");
sh.addShard("rs_shard2/localhost:27022");



use shardtest;
sh.enableSharding("shardtest");

// 100개의 더미 데이터 삽입
function dummyInput(){
    const dummy = []
    for (let i = 1; i <= 100; i++) {
          dummy.push({"userid": i})
    }
    db.test.insertMany(dummy)
}
dummyInput();

// index 설정
db.test.createIndex({ "userid": 1 });
// 설정한 index로 sharding key 설정
sh.shardCollection("shardtest.test", { "userid": 1 });

// 특정 범위로 분할
sh.splitAt("shardtest.test", { "userid": 50 });
sh.moveChunk('shardtest.test', { userid: 50 }, 'rs_shard2')

// 샤드 클러스터 상태 확인
sh.status();

// 'shardtest.test' 컬렉션의 샤딩 정보 확인
db.test.getShardDistribution();

// 새 창에서
mongosh --port 27021
use shardtest;
db.test.find().count();
```

